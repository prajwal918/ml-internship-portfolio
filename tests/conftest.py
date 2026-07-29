"""
conftest.py — pytest infrastructure only.
Mocks streamlit decorators so tests run cleanly in CI environments
without a running Streamlit session. Your project code is untouched.
"""
import sys
import unittest.mock as mock

# Patch streamlit before any project imports happen.
# @st.cache_data and @st.cache_resource become transparent pass-throughs.
_st = mock.MagicMock()
_st.cache_data = lambda **kw: (lambda f: f)
_st.cache_resource = lambda **kw: (lambda f: f)
sys.modules["streamlit"] = _st
