# 将string转换为char*

`mutable_string_data`返回的是`char*`类型，而不是`const char*`（无`const`限定符）。

``` cpp
// mutable_string_data() and as_string_data() are workarounds to improve
// the performance of writing new data to an existing string.  Unfortunately
// the methods provided by the string class are suboptimal, and using memcpy()
// is mildly annoying because it requires its pointer args to be non-NULL even
// if we ask it to copy 0 bytes.  Furthermore, string_as_array() has the
// property that it always returns NULL if its arg is the empty string, exactly
// what we want to avoid if we're using it in conjunction with memcpy()!
// With C++11, the desired memcpy() boils down to memcpy(..., &(*s)[0], size),
// where s is a string*.  Without C++11, &(*s)[0] is not guaranteed to be safe,
// so we use string_as_array(), and live with the extra logic that tests whether
// *s is empty.

// Return a pointer to mutable characters underlying the given string.  The
// return value is valid until the next time the string is resized.  We
// trust the caller to treat the return value as an array of length s->size().
inline char* mutable_string_data(string* s) {
#ifdef LANG_CXX11
  // This should be simpler & faster than string_as_array() because the latter
  // is guaranteed to return NULL when *s is empty, so it has to check for that.
  return &(*s)[0];
#else
  return string_as_array(s);
#endif
}

// as_string_data(s) is equivalent to
//  ({ char* p = mutable_string_data(s); make_pair(p, p != NULL); })
// Sometimes it's faster: in some scenarios p cannot be NULL, and then the
// code can avoid that check.
inline std::pair<char*, bool> as_string_data(string* s) {
  char *p = mutable_string_data(s);
#ifdef LANG_CXX11
  return std::make_pair(p, true);
#else
  return make_pair(p, p != NULL);
#endif
}
```