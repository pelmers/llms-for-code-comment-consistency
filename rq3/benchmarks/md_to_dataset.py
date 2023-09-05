#!/usr/bin/env python

# Usage: md_to_dataset.py <path to markdown file>
# Reads benchmark examples in markdown format from file and outputs a json data file on stdout

# markdown example entry:
'''
###### 23 ######
URL: https://github.com/gonum/plot/commit/12787dd210cb
Review: Fix comments.
Old version:
// LocScale is a scale function for a log-scale axis.
func LogScale(min, max, x float64) float64 {
        logMin := log(min)
        return (log(x) - logMin) / (log(max) - logMin)
}
New version:
// LocScale can be used as the value of an Axis.Scale function to
// set the axis to a log scale.
func LogScale(min, max, x float64) float64 {
        logMin := log(min)
        return (log(x) - logMin) / (log(max) - logMin)
}
'''
# Each of these entries turns into two entries in json: one for the old (label=1) and one for the new (label=0)
'''
[
    {
        "type": "summary",
        "id": "https://github.com/gonum/plot/commit/12787dd210cb#old",
        "old_code_raw": "func LogScale(min, max, x float64) float64 {\n\tlogMin := log(min)\n\treturn (log(x) - logMin) / (log(max) - logMin)\n}",
        "old_comment_raw": "// LocScale is a scale function for a log-scale axis.",
        "new_code_raw": "func LogScale(min, max, x float64) float64 {\n\tlogMin := log(min)\n\treturn (log(x) - logMin) / (log(max) - logMin)\n}",
        "new_comment_raw": "// LocScale is a scale function for a log-scale axis.",
        "label": 1
    },
    {
        "type": "summary",
        "id": "https://github.com/gonum/plot/commit/12787dd210cb#new",
        "old_code_raw": "func LogScale(min, max, x float64) float64 {\n\tlogMin := log(min)\n\treturn (log(x) - logMin) / (log(max) - logMin)\n}",
        "old_comment_raw": "// LocScale can be used as the value of an Axis.Scale function to\n// set the axis to a log scale.",
        "new_code_raw": "func LogScale(min, max, x float64) float64 {\n\tlogMin := log(min)\n\treturn (log(x) - logMin) / (log(max) - logMin)\n}",
        "new_comment_raw": "// LocScale can be used as the value of an Axis.Scale function to\n// set the axis to a log scale.",
        "label": 0
    }
]
'''


import os, sys, re


def parse_code(extension, code_text):
    '''
    Parse code and return code and comment
    '''
    from parsers import parse_file
    # encode code text to bytes
    if extension == ".java":
        # Java code needs to be in a class
        code_text = "class A {\n" + code_text + "\n}"
    result = parse_file(extension, code_text.encode('utf-8'))
    return result[0]['code_raw'], result[0]['comment_summary']


def main():
    # Read input file
    input_filename = sys.argv[1]
    with open(input_filename, 'r') as f:
        full_md_text = f.read()
    extension = ".go" if "go" in input_filename else ".java" if "java" in input_filename else ".py" if "py" in input_filename else ".ts"
    # Change directory to script folder
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append('../lib')
    # Split full md text by ###### <number> ######
    md_examples = full_md_text.split('###### ')[1:]
    md_jsons = []
    failed_jsons = []
    # Process each example
    for md_example in md_examples:
        # First line is the separator, so skip it
        md_example = md_example.split('\n', 1)[1]
        md_example_lower = md_example.lower()
        # Use regex to parse URL:<url>
        url = re.search('URL:\s?(.*)', md_example).group(1)
        # Join all text between Old Version: and New Version: into one string
        old_version_start_index = md_example_lower.find('old version:') + len('Old Version:')
        old_version_end_index = md_example_lower.find('new version:')
        old_version_text = md_example[old_version_start_index:old_version_end_index].strip()
        try:
            old_code, old_comment = parse_code(extension, old_version_text)
        except:
            sys.stderr.write("Error parsing code in example, complete this manually: " + url + "\n")
            failed_jsons.append({
                'type': 'summary',
                'id': url + '#old',
                'old_code_raw': '',
                'old_comment_raw': '',
                'new_code_raw': '',
                'new_comment_raw': '',
                'label': 1
            })
            failed_jsons.append({
                'type': 'summary',
                'id': url + '#new',
                'old_code_raw': '',
                'old_comment_raw': '',
                'new_code_raw': '',
                'new_comment_raw': '',
                'label': 0
            })
            continue

        new_version_start_index = md_example_lower.find('new version:') + len('New Version:')
        new_version_text = md_example[new_version_start_index:].strip()
        new_code, new_comment = parse_code(extension, new_version_text)

        # Parse text into comment and code

        # Create example json
        example_json = {
            'type': 'summary',
            'id': url + '#old',
            'old_code_raw': old_code,
            'old_comment_raw': old_comment,
            'new_code_raw': old_code,
            'new_comment_raw': old_comment,
            'label': 1
        }
        md_jsons.append(example_json)
        # Create example json
        example_json = {
            'type': 'summary',
            'id': url + '#new',
            'old_code_raw': new_code,
            'old_comment_raw': new_comment,
            'new_code_raw': new_code,
            'new_comment_raw': new_comment,
            'label': 0
        }
        md_jsons.append(example_json)

    # Print jsons as json
    import json
    print(json.dumps(md_jsons + failed_jsons, indent=4))
    sys.stderr.write("Total examples: " + str(len(md_jsons)) + "\n")


if __name__ == '__main__':
    main()