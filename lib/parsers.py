import subprocess
import os
import sys
import time

from tree_sitter import Language, Parser

from tqdm import tqdm

from pydriller import Repository, ModificationType
import difflib
import datetime

import re

dirname = os.path.abspath(os.path.dirname(__file__))
r = lambda p: os.path.join(dirname, *p.split("/"))

MAX_FILES_PER_COMMIT = 2000

def build_shared_library():
    # Clone the tree sitter vendor packages to the vendor directory
    # First make the vendor dir if it doesn't exist
    if not os.path.exists(r("vendor")):
        os.mkdir(r("vendor"))

    languages = [
        "tree-sitter-python",
        "tree-sitter-java",
        "tree-sitter-go",
        "tree-sitter-rust",
        "tree-sitter-javascript",
        "tree-sitter-typescript",
    ]
    for lang in languages:
        if not os.path.exists(r(f"vendor/{lang}")):
            subprocess.run(
                [
                    "git",
                    "clone",
                    f"https://github.com/tree-sitter/{lang}",
                    r(f"vendor/{lang}"),
                    "--depth",
                    "1",
                ]
            )
    build_paths = [r("vendor/{}".format(lang)) for lang in languages if lang != "tree-sitter-typescript"]
    # Except typescript which is in a subdirectory
    build_paths.append(r("vendor/tree-sitter-typescript/typescript"))
    Language.build_library(
        r("build/my-languages.so"), build_paths
    )


def get_parents(node):
    """
    Return list of all parents of node by walking parent pointers
    """
    parents = []
    while node.parent is not None:
        parents.append(node.parent)
        node = node.parent
    return parents


def trim_docstring_pep257(docstring):
    if not docstring:
        return ""
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    # Return a single string:
    return "\n".join(trimmed)


def trim_leading_indent_by_last_line(code):
    # Because of the way tree-sitter gives back spans, char 0 of code is the first char of the method.
    # If we want to dedent the method correctly we should use the indent level of the last line,
    # which is probably the closing brace for Java
    lines = code.expandtabs().splitlines()
    if len(lines) <= 1:
        return code
    last_line = lines[-1]
    leading_indent_amount = len(last_line) - len(last_line.lstrip())
    # special case for comments, if last line is */ then subtract 1 from the leading indent
    if last_line.strip() == "*/":
        leading_indent_amount -= 1
    dedented_lines = [lines[0]]
    for line in lines[1:]:
        if line[:leading_indent_amount].isspace():
            dedented_lines.append(line[leading_indent_amount:])
        else:
            dedented_lines.append(line)
    return "\n".join(dedented_lines)


def get_javadoc_summary(comment_raw):
    xml_regex = re.compile(r"<[^<]+>")
    special_tags = ['{', '}', '@code', '@docRoot', '@inheritDoc', '@link', '@linkplain', '@value', '@see']
    comment_no_tags = xml_regex.sub("", comment_raw)
    for t in special_tags:
        comment_no_tags = comment_no_tags.replace(t, '')
    # Parse the comment summary, based on Javadoc format: https://www.oracle.com/technical-resources/articles/java/javadoc-tool.html#descriptions
    comment_summary = ""
    comment_regex = re.compile(r"(^\s*/*\*+/?)|(\*/$)")
    for line in comment_no_tags.splitlines():
        # First remove the /**, *, or */ for every line with regex
        line = line.strip()
        line = comment_regex.sub("", line).strip()
        if line == "" or line == "<br>" or line == "<br />":
            if comment_summary == "":
                continue
            else:
                break
        # if the first word starts with @, like @param or @return, then break
        first_word = line.split(" ")[0]
        first_word_is_directive = first_word.startswith("@param") or first_word.startswith("@return") or first_word.startswith("@throws")
        if first_word_is_directive:
            break
        if ". " in line:
            part = line[: line.find(". ") + 1]
            comment_summary += part if not comment_summary else " " + part
            break
        elif line[-1] == ".":
            comment_summary += line if not comment_summary else " " + line
            break
        else:
            comment_summary += line if not comment_summary else " " + line
    return comment_summary


def parse_entire_directory(directory, file_extension):
    """
    Parse all files with the given file extension in the supplied directory.
    The parser is determined from the extension.
    Supported extensions: .java, .py, .go, .rs, .js

    Return a list of examples, where each example is a dictionary with the fields:
    "code_raw",
    "comment_raw",
    "comment_summary",
    "code_start_line",
    "code_start_char",
    "comment_start_line",
    "comment_start_char",
    "qualified_name",
    """
    build_shared_library()
    assert file_extension in [".java", ".py", ".go", ".rs", ".js", ".ts"]
    results = []
    found_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(file_extension):
                path = os.path.join(root, file)
                found_files.append(path)
    for path in tqdm(found_files):
        # path relative to the directory root
        rel_path = os.path.relpath(path, directory)
        # Add the relative path to all the examples in the file
        try:
            examples = parse_file(path)
        except Exception as e:
            print("Error parsing file: ", path)
            print(e)
            continue
        # If the file is in a git repo, then record the number of seconds since the last commit to the file
        try:
            log_result = subprocess.run(["git", "log", "-1", "--format=%ct", path], stdout=subprocess.PIPE, cwd=directory)
            commit_timestamp = int(log_result.stdout)
            seconds_since_last_commit = time.time() - commit_timestamp
        except Exception as e:
            seconds_since_last_commit = 1e9
        for example in examples:
            example["path"] = rel_path
            example["seconds_since_last_commit"] = seconds_since_last_commit
            results.append(example)
    return results


def parse_file(path, file_contents=None):
    """
    Return the same thing as entire_directory but just for one file
    If contents is not given, we read path from disk.
    """
    if not file_contents:
        with open(path, "r") as f:
            file_contents = bytes(f.read(), "utf8")
    # If file is > 500kb, then skip
    if len(file_contents) > 500000:
        return []
    if path.endswith(".py"):
        return parse_file_python(path, file_contents)
    elif path.endswith(".java"):
        return parse_file_java(path, file_contents)
    elif path.endswith(".go"):
        return parse_file_go(path, file_contents)
    elif path.endswith(".js") or path.endswith(".ts"):
        return parse_file_js(path, file_contents)


def create_dataset_example_object(additional_fields, before_item, after_item):
    # previous dataset's keys: ['id', 'label', 'comment_type', 'old_comment_raw', 'old_comment_subtokens', 'new_comment_raw', 'new_comment_subtokens', 'span_minimal_diff_comment_subtokens', 'old_code_raw', 'old_code_subtokens', 'new_code_raw', 'new_code_subtokens', 'span_diff_code_subtokens', 'token_diff_code_subtokens']
    if 'end_newlines' in before_item and 'end_newlines' in after_item:
        additional_fields["old_end_newlines"] = before_item["end_newlines"]
        additional_fields["new_end_newlines"] = after_item["end_newlines"]
    if 'kind' in before_item:
        additional_fields["kind"] = before_item["kind"]
    return {
        "old_entire_comment_raw": before_item["comment_raw"],
        "new_entire_comment_raw": after_item["comment_raw"],
        "old_code_raw": before_item["code_raw"],
        "new_code_raw": after_item["code_raw"],
        "old_comment_raw": before_item["comment_summary"],
        "new_comment_raw": after_item["comment_summary"],
        "old_code_start_line": before_item["code_start_line"],
        "old_code_start_char": before_item["code_start_char"],
        "new_code_start_line": after_item["code_start_line"],
        "new_code_start_char": after_item["code_start_char"],
        "old_comment_start_line": before_item["comment_start_line"],
        "old_comment_start_char": before_item["comment_start_char"],
        "new_comment_start_line": after_item["comment_start_line"],
        "new_comment_start_char": after_item["comment_start_char"],
        "qualified_name": before_item["qualified_name"],
        "type": 'summary',
        **additional_fields,
    }


def get_language_and_parser(lang):
    language = Language(r("build/my-languages.so"), lang)
    parser = Parser()
    parser.set_language(language)
    return language, parser


def parse_file_python(path, code):
    # If "generated" appears in the first 100 characters of the code, then mark it as maybe generated
    maybe_generated = b"generated" in code[:100]
    language, parser = get_language_and_parser('python')
    tree = parser.parse(code)
    node = tree.root_node
    # Based on example at https://pypi.org/project/tree-sitter-languages/
    # Parser query that finds all nodes for function doc strings (string immediately following def)
    function_doc_pattern = """
        (function_definition
            body: (block . (expression_statement (string) @function_doc_str)))
    """
    function_doc_query = language.query(function_doc_pattern)
    doc_strs = function_doc_query.captures(node)
    # Get the code contents for each point as a result
    results = []
    for (node, tag) in doc_strs:
        # Go up the parents of the doc comment to the function definition
        func_node = node.parent.parent.parent
        assert func_node.type == "function_definition"
        # If any parent of func_node is also a function, skip because it is a nested function
        if any([p.type == "function_definition" for p in get_parents(func_node)]):
            continue
        # If the func is in a class, then it is a method, so record the class name
        class_name = next(
            (
                p.named_children[0].text.decode("utf8")
                for p in get_parents(func_node)
                if p.type == "class_definition"
            ),
            None,
        )
        # We can qualify names in python as module.class.method
        module_name = path.split("/")[-1].split(".")[0]
        function_name = func_node.named_children[0].text.decode("utf8")
        if class_name is not None:
            qualified_name = f"{module_name}.{class_name}.{function_name}"
        else:
            qualified_name = f"{module_name}.{function_name}"
        # For python, include the white space after the "def" expression in the comment
        punctuation = [n for n in func_node.children if n.type == ":"][-1]
        comment_raw = node.text.decode("utf8")
        comment_quote_char = comment_raw[-1]
        comment_quote_start = comment_raw.find(comment_quote_char)
        comment_docstring = trim_docstring_pep257(
            comment_raw[comment_quote_start + 3 : -3]
        )
        comment_lines = comment_docstring.splitlines()
        # Summary is to the first period or until the first empty line or until a line with a colon or @ sign in the first 20 characters
        comment_summary = ""
        for line in comment_lines:
            first_colon = line.strip().find(":")
            first_at = line.strip().find("@")
            if line.strip() == "" or (first_colon != -1 and first_colon < 20) or (first_at != -1 and first_at < 20):
                if comment_summary == "":
                    continue
                else:
                    break
            if ". " in line:
                part = line[: line.find(". ") + 1]
                comment_summary += part if not comment_summary else " " + part
                break
            elif line[-1] == ".":
                comment_summary += line if not comment_summary else " " + line
                break
            else:
                # Haven't seen a period, so don't break
                comment_summary += line if not comment_summary else " " + line
        # The function code is the span of this node, but with the doc comment removed
        # Strip the leading spaces of the docstring
        code_raw = (
            code[func_node.start_byte : node.start_byte].decode("utf8").rstrip() +
            code[node.end_byte : func_node.end_byte].decode("utf8")
        )
        results.append(
            {
                "code_raw": code_raw,
                # Comment has a leading newline so we can strip that
                "comment_raw": comment_raw.strip(),
                "comment_summary": comment_summary.strip(),
                "code_start_line": func_node.start_point[0],
                "code_start_char": func_node.start_point[1],
                "comment_start_line": node.start_point[0],
                "comment_start_char": node.start_point[1],
                "qualified_name": qualified_name,
                "maybe_generated": maybe_generated,
            }
        )
    return results


def parse_file_java(path, code):
    # If "generated" appears in the first 100 characters of the code, then mark it as maybe generated
    maybe_generated = b"generated" in code[:100]
    language, parser = get_language_and_parser('java')
    tree = parser.parse(code)
    node = tree.root_node
    # Tree query that finds all nodes for method javadoc comments
    javadoc_pattern = """
    (
        (block_comment) @java_comment
        (method_declaration)
    )
    """
    javadoc_query = language.query(javadoc_pattern)
    javadoc_captures = javadoc_query.captures(node)
    first_by_type = lambda children, typ: next(
        (c for c in children if c.type == typ), None
    )
    results = []
    for (node, tag) in javadoc_captures:
        method_node = node.next_sibling
        if method_node.type != "method_declaration":
            # Could be a nested class or interface or something
            # Why doesn't the query above avoid this?
            # https://tree-sitter.github.io/tree-sitter/using-parsers#pattern-matching-with-queries
            # mentioned on https://www.bearer.com/blog/tips-for-using-tree-sitter-queries,
            # you get every possible combination of nodes
            continue
        # If the method is abstract then skip it
        modifiers = first_by_type(method_node.children, "modifiers")
        if modifiers and b'abstract' in modifiers.text:
            continue
        class_node = method_node.parent.parent
        if class_node.type != "class_declaration":
            continue
        comment_raw = trim_leading_indent_by_last_line(node.text.decode("utf8"))
        comment_summary = get_javadoc_summary(comment_raw)
        code_raw = trim_leading_indent_by_last_line(method_node.text.decode("utf8"))
        method_name_node = first_by_type(
            method_node.named_children, "identifier"
        )
        method_name = method_name_node.text.decode("utf8")
        class_name = first_by_type(class_node.named_children, "identifier").text.decode(
            "utf8"
        )
        params_node = first_by_type(method_node.named_children, "formal_parameters")
        # the qualified name of a method needs to include the types of the parameters
        # because of overloading
        qualified_name = f'{class_name}.{method_name}'
        # One last thing we want to know for Java: how many empty newlines are there after the method?
        method_end_line = method_node.end_point[0]
        # There is one by default
        end_newlines = 1
        if method_node.next_sibling:
            end_newlines += method_node.next_sibling.start_point[0] - method_end_line - 1

        results.append(
            {
                "code_raw": code_raw,
                "comment_raw": comment_raw,
                "comment_summary": comment_summary,
                "code_start_line": method_node.start_point[0],
                "code_start_char": method_node.start_point[1],
                "comment_start_line": node.start_point[0],
                "comment_start_char": node.start_point[1],
                "qualified_name": qualified_name,
                "parameters": params_node.text.decode("utf8"),
                "end_newlines": end_newlines,
                "maybe_generated": maybe_generated,
            }
        )
    return results


def parse_file_go(path, code):
    # If "generated" appears in the first 100 characters of the code, then mark it as maybe generated
    maybe_generated = b"generated" in code[:100]
    language, parser = get_language_and_parser('go')
    tree = parser.parse(code)
    node = tree.root_node
    # Godoc comments are a series of single line comments
    # We write a pattern to find all the function/method declarations, then programmatically
    # collect previous siblings that are comments on consecutive lines
    gofunc_p = "([(function_declaration) @fd (method_declaration) @md])"
    gofunc_q = language.query(gofunc_p)
    first_by_types = lambda children, typs: next(
        (c for c in children if c.type in typs), None
    )
    godoc_captures = gofunc_q.captures(node)
    results = []
    for (func_node, tag) in godoc_captures:
        # Walk up the siblings to collect comments which are on consecutive lines from this declaration
        comment_nodes = []
        cur_line = func_node.start_point[0]
        prev_node = func_node.prev_sibling
        while prev_node and prev_node.type == "comment" and prev_node.start_point[0] == cur_line - 1:
            comment_nodes.append(prev_node)
            cur_line -= 1
            prev_node = prev_node.prev_sibling
        if not comment_nodes:
            continue
        # Reverse list of comment nodes because we collected them in reverse order
        comment_nodes.reverse()
        comment_raw = trim_leading_indent_by_last_line("\n".join([n.text.decode("utf8") for n in comment_nodes]))
        comment_summary = get_godoc_summary(comment_raw)

        if tag == "fd":
            qualified_name = first_by_types(func_node.named_children, ['identifier']).text.decode()
        else:
            receiver_node = first_by_types(func_node.named_children, ['parameter_list'])
            struct_decl = first_by_types(receiver_node.named_children, ['parameter_declaration'])
            # The struct name may be under a pointer or a value
            struct_name = first_by_types(struct_decl.named_children, ['pointer_type', 'type_identifier', 'generic_type']).text.decode()
            method_name = first_by_types(func_node.named_children, ['field_identifier']).text.decode()
            qualified_name = f'{struct_name}.{method_name}'
        code_raw = trim_leading_indent_by_last_line(func_node.text.decode("utf8"))
        results.append(
            {
                "code_raw": code_raw,
                "comment_raw": comment_raw,
                "comment_summary": comment_summary,
                "code_start_line": func_node.start_point[0],
                "code_start_char": func_node.start_point[1],
                "comment_start_line": comment_nodes[0].start_point[0],
                "comment_start_char": comment_nodes[0].start_point[1],
                "qualified_name": qualified_name,
                "maybe_generated": maybe_generated,
            }
        )
    return results


def get_godoc_summary(comment_raw):
    comment_lines = comment_raw.splitlines()
    # Summary is to the first period or until the first empty line or until a line with a colon or @ sign in the first 20 characters
    comment_summary = ""
    for line in comment_lines:
        # Remove all leading slashes and whitespace
        line = re.sub(r"^[\s/]*", "", line)
        if line == "":
            if comment_summary == "":
                continue
            else:
                break
        if ". " in line:
            part = line[: line.find(". ") + 1]
            comment_summary += part if not comment_summary else " " + part
            break
        elif line[-1] == ".":
            comment_summary += line if not comment_summary else " " + line
            break
        else:
            # Haven't seen a period, so don't break
            comment_summary += line if not comment_summary else " " + line
    return comment_summary


def parse_file_js(path, code):
    # Skip parsing if the path contains node_modules, dist, or build
    path_components = path.split('/')
    if any([x in path_components for x in ["node_modules", "dist", "build"]]):
        return []
    # If any line of code is longer than 390 characters, then skip parsing because it's probably a minified file
    if any([len(line) > 390 for line in code.splitlines()]):
        return []
    # If "generated" appears in the first 100 characters of the code, then mark it as maybe generated
    maybe_generated = b"generated" in code[:100]
    language_name = 'javascript' if path.endswith('.js') else 'typescript'
    language, parser = get_language_and_parser(language_name)
    tree = parser.parse(code)
    node = tree.root_node
    # Similar to Java, look for block comments immediately preceding a function/method declaration
    jsdoc_p = """
    (
        (comment) @jsdoc
        [(function_declaration) (method_definition)]
    )
    """
    jsdoc_q = language.query(jsdoc_p)
    first_by_types = lambda children, typs: next(
        (c for c in children if c.type in typs), None
    )
    results = []
    def func_name(func_node):
        if func_node.type == "function_declaration":
            func_name_node = first_by_types(func_node.named_children, ['identifier'])
        else:
            func_name_node = first_by_types(func_node.named_children, ['property_identifier'])
        if not func_name_node:
            return None
        return func_name_node.text.decode("utf8")

    for (comment_node, tag) in jsdoc_q.captures(node):
        # We are looking for: not-nested, block comment
        if not comment_node.text.startswith(b"/*"):
            continue
        if first_by_types(get_parents(comment_node), ["function_declaration", "method_definition"]):
            continue
        func_node = comment_node.next_named_sibling
        if not func_node:
            continue
        if func_node.start_point[0] != comment_node.end_point[0] + 1:
            continue
        if func_node.type not in ["function_declaration", "method_definition"]:
            continue
        comment_raw = trim_leading_indent_by_last_line(comment_node.text.decode("utf8"))
        comment_summary = get_javadoc_summary(comment_raw)
        qualified_name = func_name(func_node)
        if not qualified_name:
            continue
        code_raw = trim_leading_indent_by_last_line(func_node.text.decode("utf8"))
        results.append(
            {
                "code_raw": code_raw,
                "comment_raw": comment_raw,
                "comment_summary": comment_summary,
                "code_start_line": func_node.start_point[0],
                "code_start_char": func_node.start_point[1],
                "comment_start_line": comment_node.start_point[0],
                "comment_start_char": comment_node.start_point[1],
                "qualified_name": qualified_name,
                "maybe_generated": maybe_generated,
                "kind": tag,
            }
        )
    # There are simply too few comments that follow JS Doc format, so we will also use the Go approach
    # Find all the function/method declarations, look for comments immediately preceding them
    jsfunc_p = "([(function_declaration) @fd (method_definition) @md])"
    jsfunc_q = language.query(jsfunc_p)
    for (func_node, tag) in jsfunc_q.captures(node):
        if first_by_types(get_parents(func_node), ["function_declaration", "method_definition"]):
            continue
        # Collect comments immediately preceding the function/method declaration
        comment_nodes = []
        cur_line = func_node.start_point[0]
        prev_node = func_node.prev_sibling
        while prev_node and prev_node.type == "comment" and prev_node.start_point[0] == cur_line - 1:
            comment_nodes.append(prev_node)
            cur_line -= 1
            prev_node = prev_node.prev_sibling
        if not comment_nodes:
            continue
        # Reverse list of comment nodes because we collected them in reverse order
        comment_nodes.reverse()
        comment_raw = trim_leading_indent_by_last_line("\n".join([n.text.decode("utf8") for n in comment_nodes]))
        # The other path already covers /* comments
        if not comment_raw.startswith("//"):
            continue
        comment_summary = get_godoc_summary(comment_raw)
        qualified_name = func_name(func_node)
        if not qualified_name:
            continue
        code_raw = trim_leading_indent_by_last_line(func_node.text.decode("utf8"))
        results.append(
            {
                "code_raw": code_raw,
                "comment_raw": comment_raw,
                "comment_summary": comment_summary,
                "code_start_line": func_node.start_point[0],
                "code_start_char": func_node.start_point[1],
                "comment_start_line": comment_nodes[0].start_point[0],
                "comment_start_char": comment_nodes[0].start_point[1],
                "qualified_name": qualified_name,
                "maybe_generated": maybe_generated,
                "kind": tag,
            }
        )

    return results


# regex that matches whitespace or any of ",", ".", "!", "?", "'", '"', "`", ";", ":"
fuzzy_match_re = re.compile(r"(\s|,|\.|!|\?|'|\"|`|;|:)+")
def comments_fuzzy_equal(before, after, fuzziness=0):
    '''
    Consider comments equal if they are the same or only differ by punctuation
    if fuzziness is given, then that is the number of non-punctuation edits allowed before we say they are different
    '''
    before = before.lower()
    after = after.lower()
    if fuzziness == 0:
        return fuzzy_match_re.sub("", before) == fuzzy_match_re.sub("", after)
    # Use difflib to check the shortest edit, if it is just punctuation or spaces, then they are equal
    # Punctuation characters: ,.!?'"`
    diff = difflib.ndiff(before, after)
    # First char is +, -, ? or space
    diff = [d for d in diff if d[0] != " "]
    if len(diff) == 0:
        return True
    fuz = 0
    for d in diff:
        if d[2] not in {",", ".", "!", "?", "'", '"', "`", ";", ":", " "}:
            fuz += 1
            if fuz > fuzziness:
                return False
    return True


def create_labeled_dataset_from_git_repo(directory, file_extension, time_limit=None, fixed_comments_only=False, days_of_commits=None):
    """
    Go through the git history of the given directory, get examples of code and comment
    labeled by the assumption that:
    if the code and comment both changed,
    then then the old comment is incorrect and the new comment is correct
    """
    build_shared_library()
    results = []
    try:
        # Get the repo Url from git origin remote
        repo_url = (
            subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=directory,
                stdout=subprocess.PIPE,
            )
            .stdout.decode("utf8")
            .strip()
        )
        # For each commit in the git history of the directory,
        # parse every modified file of given file extension before and after the commit
        # for any qualified name that has both code and comment changed by the commit,
        # label the old comment with new code as 1 (inconsistent), and the new comment with new code as 0 (consistent)
        branch_name = (
            subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=directory,
                stdout=subprocess.PIPE,
            )
            .stdout.decode("utf8")
            .strip()
        )
        commit_count = int(
            subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=directory,
                stdout=subprocess.PIPE,
            )
            .stdout.decode("utf8")
            .strip()
        )
    except Exception as e:
        # If any of the above fails, return an empty list and print a message
        print(
            f"Could not get git history for {directory} due to {e}. Skipping this directory."
        )
        return []

    def group_by_name(items):
        '''Group each item by qualified name as a dictionary with list values'''
        d = {}
        for item in items:
            if item["qualified_name"] not in d:
                d[item["qualified_name"]] = []
            d[item["qualified_name"]].append(item)
        return d

    start_time = time.time()
    repo = Repository(directory, only_in_branch=branch_name)
    progress = tqdm(repo.traverse_commits(), total=commit_count)
    for commit in progress:
        # If days_of_commits is given and the commit is older than that many days, then skip
        if days_of_commits is not None and (datetime.datetime.now() - commit.committer_date.replace(tzinfo=None)).days > days_of_commits:
            continue
        progress.set_postfix_str(f"({len(results)})", refresh=False)
        # Docs at https://pydriller.readthedocs.io/en/latest/tutorial.html
        try:
            relevant_files = [
                f
                for f in commit.modified_files
                if f.change_type == ModificationType.MODIFY
                and f.new_path.endswith(file_extension)
                and f.source_code is not None
                and f.source_code_before is not None
            ][:MAX_FILES_PER_COMMIT]
        except Exception:
            continue

        if time_limit is not None and time.time() - start_time > time_limit:
            break
        for f in relevant_files:
            after_items = group_by_name(parse_file(f.new_path, bytes(f.source_code, "utf8")))
            before_items = group_by_name(
                parse_file(f.new_path, bytes(f.source_code_before, "utf8"))
            )
            for name in before_items:
                if name not in after_items:
                    continue
                # If the number of items with the name is different, we cannot match them up
                if len(after_items[name]) != len(before_items[name]):
                    continue
                for (after_item, before_item) in zip(after_items[name], before_items[name]):
                    # If either comment summary is empty, skip
                    if not after_item["comment_summary"] or not before_item["comment_summary"]:
                        continue
                    # If either comment or code contains non-ascii characters, skip
                    if (
                        not after_item["comment_summary"].isascii() or
                        not before_item["comment_summary"].isascii() or
                        not after_item["code_raw"].isascii() or
                        not before_item["code_raw"].isascii()
                    ):
                        continue
                    code_changed = not comments_fuzzy_equal(before_item["code_raw"], after_item["code_raw"])
                    # Previous work's definition of labeling:
                    # If the code changed, then the label is 0 if the comment is the same, and 1 if the comment changed
                    # it refers to whether the OLD comment matches the NEW code
                    # If the "change" is only whitespace at the beginning of lines, then skip
                    # previous dataset's keys: ['id', 'label', 'comment_type', 'old_comment_raw', 'old_comment_subtokens', 'new_comment_raw', 'new_comment_subtokens', 'span_minimal_diff_comment_subtokens', 'old_code_raw', 'old_code_subtokens', 'new_code_raw', 'new_code_subtokens', 'span_diff_code_subtokens', 'token_diff_code_subtokens']
                    label = (
                        0
                        if comments_fuzzy_equal(before_item["comment_summary"], after_item["comment_summary"])
                        else 1
                    )
                    # If only looking for fixed comments, then skip code changes and unchanged comments
                    if fixed_comments_only:
                        if label == 0 or code_changed:
                            continue
                    else:
                        # Otherwise skip if the code was not changed
                        if not code_changed:
                            continue
                    results.append(
                        create_dataset_example_object(
                            {
                                "label": label,
                                "repo_url": repo_url,
                                "path": f.new_path,
                                "commit": commit.hash,
                                "commit_msg": commit.msg,
                                "id": f'{"/".join(repo_url.split("/")[-2:])}#{len(results)}'
                            },
                            before_item,
                            after_item,
                        )
                    )

    return results


if __name__ == "__main__":
    # When called directly, clone the repo given in the first argument from github (like pelmers/text-rewriter)
    # save it to a temporary directory and parse its contents
    # using the extension in the second argument
    # and print the results as JSON
    import json
    import sys

    repo = f"https://github.com/{sys.argv[1]}.git"
    extension = sys.argv[2]
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(["git", "clone", repo, tmpdir])
        results = parse_entire_directory(tmpdir, extension)
        # results = create_labeled_dataset_from_git_repo(tmpdir, extension)

        print(len(json.dumps(results)))
