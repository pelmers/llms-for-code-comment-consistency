These are the examples from the python sample set of 16 cases,
to be validated against my "volunteers'" annotations.

(completed)

######

URL: https://github.com/astropy/astropy/pull/3067#discussion_r19736856

Review: Just update this docstring and this should be fine.  Something more along the lines of what it actually tests.  For example
```
"""
Returns `True` if ``HDUList.index_of(item)`` succeeds.
"""
```

Old Version:
def __contains__(self, item):
    """
    Used by the 'in' operator
    """
    try:
        self.index_of(item)
        return True
    except KeyError:
        return False

New Version:
def __contains__(self, item):
    """
    Returns `True` if HDUList.index_of(item) succeeds.
    """
    try:
        self.index_of(item)
        return True
    except KeyError:
        return False

######
URL: https://github.com/pretix/pretix/pull/334#discussion_r89618197

Review: Please update the docstring here, aswell.

Old Version:
def is_allowed(self, request: HttpRequest) -> bool:
    """
    You can use this method to disable this payment provider for certain groups
    of users, products or other criteria. If this method returns ``False``, the
    user will not be able to select this payment method. This will only be called
    during checkout, not on retrying.

    The default implementation always returns ``True``.
    """
    return self._is_still_available()

New Version:
def is_allowed(self, request: HttpRequest) -> bool:
    """
    You can use this method to disable this payment provider for certain groups
    of users, products or other criteria. If this method returns ``False``, the
    user will not be able to select this payment method. This will only be called
    during checkout, not on retrying.

    The default implementation checks for the _availability_date setting to be either unset or in the future.
    """
    return self._is_still_available()

######
URL: https://github.com/matplotlib/matplotlib/pull/5942#discussion_r51233494

Review: Should the docstring be updated with this change

Old Version:
def option_scale_image(self):
    """
    agg backend support arbitrary scaling of image.
    """
    return True

New Version:
def option_scale_image(self):
    """
    agg backend doesn't support arbitrary scaling of image.
    """
    return False
