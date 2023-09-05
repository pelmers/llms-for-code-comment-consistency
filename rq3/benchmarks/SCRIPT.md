Preface
=======

Each raw example candidate will look like this:
{
    "html_url": "https://github.com/open-mmlab/mmdetection/pull/160#discussion_r244680239",
    "path": "mmdet/models/utils/norm.py",
    "line": 30.0,
    "body": "The docstring needs to be updated.",
    "user": "hellock",
    "diff_hunk": "@@ -1,17 +1,48 @@\n import torch.nn as nn\n \n-norm_cfg = {'BN': nn.BatchNorm2d, 'SyncBN': None, 'GN': None}\n \n+norm_cfg = {\n+    # format: layer_type: (abbreation, module)\n+    'BN': ('bn', nn.BatchNorm2d),\n+    'SyncBN': ('bn', None),\n+    'GN': ('gn', nn.GroupNorm),\n+    # and potentially 'SN'\n+}\n \n-def build_norm_layer(cfg, num_features):\n+\n+def build_norm_layer(cfg, num_features, postfix=''):\n+    \"\"\"\n+    cfg should contain:\n+        type (str): identify norm layer type.\n+        layer args: args needed to instantiate a norm layer.\n+        frozen (bool): [optional] whether stop gradient updates\n+            of norm layer, it is helpful to set frozen mode\n+            in backbone's norms.\n+    \"\"\"",
    "author_association": "MEMBER",
    "commit_id": "2b906db9d69845b7c5a9c3c4822de7eac5c3557e",
    "id": 244680239,
    "repo": "open-mmlab/mmdetection",
    "extension": ".py"
}

For the benchmark data, we want:
Old comment + code (the version that was marked wrong)
New comment + code (the version after the author corrected it)
The content of the PR comment that signals a change was required

=========
Procedure
=========
1. Click the html_url in the example
2. On the GitHub review page, confirm the following:
   a) The code in question is in fact a documentation comment for a FUNCTION/METHOD (not a CLASS!)
   b) The PR comment asks the author to change/update their work
   c) The author did in fact update their work in response to the comment
   d) The update occurs in the comment SUMMARY lines, not only in a PARAMETER or RETURN line
3. If all pass, then create the benchmark example in the format below.
4. Use 6 pounds ###### to divide examples

Notes:
1. To see the full contents of the file beyond the highlighted region on GitHub, you can click the filename.
2. For easier copy paste, click the '...' at the top of the file divider and choose View File
3. To see changes AFTER the review, set the drop down at the top to changes from all commits
4. In case the commit was squashed, the PRE-REVIEW code is not visible on GitHub anymore.
   You can try to find it from the "diff_hunk" field of the given JSON. Otherwise just skip it.

Tip: GitHub UI sometimes scrolls you to the top of the pull request,
     which is annoying if it's a long change.
     Copy a unique string (e.g. function name) and search for it to find your place again.

===============
Example Outcome
===============
URL: https://github.com/open-mmlab/mmdetection/pull/160#discussion_r244680239

Review:
The docstring needs to be updated.

Old Version:
def build_norm_layer(cfg, num_features, postfix=''):
    """
    cfg should contain:
        type (str): identify norm layer type.
        layer args: args needed to instantiate a norm layer.
        frozen (bool): [optional] whether stop gradient updates
            of norm layer, it is helpful to set frozen mode
            in backbone's norms.
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    frozen = cfg_.pop('frozen', False)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    if frozen:
        for param in layer.parameters():
            param.requires_grad = False

    return name, layer

New Version:
def build_norm_layer(cfg, num_features, postfix=''):
    """ Build normalization layer

    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            frozen (bool): [optional] whether stop gradient updates
                of norm layer, it is helpful to set frozen mode
                in backbone's norms.
        num_features (int): number of channels from input
        postfix (int, str): appended into norm abbreation to
            create named layer.

    Returns:
        name (str): abbreation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    frozen = cfg_.pop('frozen', False)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    if frozen:
        for param in layer.parameters():
            param.requires_grad = False

    return name, layer

######