# `ZeroCopyStream`

file: `<google/io/zero_copy_stream.h>`

在编程上，没有给我很特别的感觉。主要是定义了一些protobuf内部用到的input和output stream的接口：`ZeroCopyInputStream`和`ZeroCopyOutputStream`。它的目标是实现对stream读写的尽量少的拷贝，甚至是**零拷贝**。

它主要的接口是四个函数，没有很特殊的感觉：
- `Next()`
- `Backup()`
- `Skip()`
- `ByteCount()`

它的实现有：
``` cpp
// in <google/io/zero_copy_stream_impl.h>
class LIBPROTOBUF_EXPORT FileInputStream : public ZeroCopyInputStream;
class LIBPROTOBUF_EXPORT FileOutputStream : public ZeroCopyOutputStream;
class LIBPROTOBUF_EXPORT IstreamInputStream : public ZeroCopyInputStream;
class LIBPROTOBUF_EXPORT OstreamOutputStream : public ZeroCopyOutputStream;
class LIBPROTOBUF_EXPORT ConcatenatingInputStream : public ZeroCopyInputStream;
class LIBPROTOBUF_EXPORT LimitingInputStream : public ZeroCopyInputStream;

// in <google/io/zero_copy_stream_impl_lite.h>
class LIBPROTOBUF_EXPORT ArrayInputStream : public ZeroCopyInputStream;
class LIBPROTOBUF_EXPORT ArrayOutputStream : public ZeroCopyOutputStream;
class LIBPROTOBUF_EXPORT StringOutputStream : public ZeroCopyOutputStream;
class LIBPROTOBUF_EXPORT LazyStringOutputStream : public StringOutputStream;
class LIBPROTOBUF_EXPORT CopyingInputStreamAdaptor : public ZeroCopyInputStream;
class LIBPROTOBUF_EXPORT CopyingOutputStreamAdaptor : public ZeroCopyOutputStream;
```

另外，对于`CopyingInputStreamAdaptor`和`CopyingOutputStreamAdaptor`这两个adaptor类，是可能需要拷贝的情况，在这种情况下，要分别实现`CopyingInputStream`和`CopyingOutputStream`。然后使用这两种adaptor来处理需要拷贝的情况，比如file、C stdio stream、C++ iostream等。

``` cpp
// A generic traditional input stream interface.
//
// Lots of traditional input streams (e.g. file descriptors, C stdio
// streams, and C++ iostreams) expose an interface where every read
// involves copying bytes into a buffer.  If you want to take such an
// interface and make a ZeroCopyInputStream based on it, simply implement
// CopyingInputStream and then use CopyingInputStreamAdaptor.
//
// CopyingInputStream implementations should avoid buffering if possible.
// CopyingInputStreamAdaptor does its own buffering and will read data
// in large blocks.
class LIBPROTOBUF_EXPORT CopyingInputStream {
 public:
  virtual ~CopyingInputStream();

  // Reads up to "size" bytes into the given buffer.  Returns the number of
  // bytes read.  Read() waits until at least one byte is available, or
  // returns zero if no bytes will ever become available (EOF), or -1 if a
  // permanent read error occurred.
  virtual int Read(void* buffer, int size) = 0;

  // Skips the next "count" bytes of input.  Returns the number of bytes
  // actually skipped.  This will always be exactly equal to "count" unless
  // EOF was reached or a permanent read error occurred.
  //
  // The default implementation just repeatedly calls Read() into a scratch
  // buffer.
  virtual int Skip(int count);
};

// A generic traditional output stream interface.
//
// Lots of traditional output streams (e.g. file descriptors, C stdio
// streams, and C++ iostreams) expose an interface where every write
// involves copying bytes from a buffer.  If you want to take such an
// interface and make a ZeroCopyOutputStream based on it, simply implement
// CopyingOutputStream and then use CopyingOutputStreamAdaptor.
//
// CopyingOutputStream implementations should avoid buffering if possible.
// CopyingOutputStreamAdaptor does its own buffering and will write data
// in large blocks.
class LIBPROTOBUF_EXPORT CopyingOutputStream {
 public:
  virtual ~CopyingOutputStream();

  // Writes "size" bytes from the given buffer to the output.  Returns true
  // if successful, false on a write error.
  virtual bool Write(const void* buffer, int size) = 0;
};
```

下面是`ZeroCopyInputStream`和`ZeroCopyOutputStream`的接口定义源代码。

``` cpp
// Author: kenton@google.com (Kenton Varda)
//  Based on original Protocol Buffers design by
//  Sanjay Ghemawat, Jeff Dean, and others.
//
// This file contains the ZeroCopyInputStream and ZeroCopyOutputStream
// interfaces, which represent abstract I/O streams to and from which
// protocol buffers can be read and written.  For a few simple
// implementations of these interfaces, see zero_copy_stream_impl.h.
//
// These interfaces are different from classic I/O streams in that they
// try to minimize the amount of data copying that needs to be done.
// To accomplish this, responsibility for allocating buffers is moved to
// the stream object, rather than being the responsibility of the caller.
// So, the stream can return a buffer which actually points directly into
// the final data structure where the bytes are to be stored, and the caller
// can interact directly with that buffer, eliminating an intermediate copy
// operation.
//
// As an example, consider the common case in which you are reading bytes
// from an array that is already in memory (or perhaps an mmap()ed file).
// With classic I/O streams, you would do something like:
//   char buffer[BUFFER_SIZE];
//   input->Read(buffer, BUFFER_SIZE);
//   DoSomething(buffer, BUFFER_SIZE);
// Then, the stream basically just calls memcpy() to copy the data from
// the array into your buffer.  With a ZeroCopyInputStream, you would do
// this instead:
//   const void* buffer;
//   int size;
//   input->Next(&buffer, &size);
//   DoSomething(buffer, size);
// Here, no copy is performed.  The input stream returns a pointer directly
// into the backing array, and the caller ends up reading directly from it.
//
// If you want to be able to read the old-fashion way, you can create
// a CodedInputStream or CodedOutputStream wrapping these objects and use
// their ReadRaw()/WriteRaw() methods.  These will, of course, add a copy
// step, but Coded*Stream will handle buffering so at least it will be
// reasonably efficient.
//
// ZeroCopyInputStream example:
//   // Read in a file and print its contents to stdout.
//   int fd = open("myfile", O_RDONLY);
//   ZeroCopyInputStream* input = new FileInputStream(fd);
//
//   const void* buffer;
//   int size;
//   while (input->Next(&buffer, &size)) {
//     cout.write(buffer, size);
//   }
//
//   delete input;
//   close(fd);
//
// ZeroCopyOutputStream example:
//   // Copy the contents of "infile" to "outfile", using plain read() for
//   // "infile" but a ZeroCopyOutputStream for "outfile".
//   int infd = open("infile", O_RDONLY);
//   int outfd = open("outfile", O_WRONLY);
//   ZeroCopyOutputStream* output = new FileOutputStream(outfd);
//
//   void* buffer;
//   int size;
//   while (output->Next(&buffer, &size)) {
//     int bytes = read(infd, buffer, size);
//     if (bytes < size) {
//       // Reached EOF.
//       output->BackUp(size - bytes);
//       break;
//     }
//   }
//
//   delete output;
//   close(infd);
//   close(outfd);


// Abstract interface similar to an input stream but designed to minimize
// copying.
class LIBPROTOBUF_EXPORT ZeroCopyInputStream {
 public:
  inline ZeroCopyInputStream() {}
  virtual ~ZeroCopyInputStream();

  // Obtains a chunk of data from the stream.
  //
  // Preconditions:
  // * "size" and "data" are not NULL.
  //
  // Postconditions:
  // * If the returned value is false, there is no more data to return or
  //   an error occurred.  All errors are permanent.
  // * Otherwise, "size" points to the actual number of bytes read and "data"
  //   points to a pointer to a buffer containing these bytes.
  // * Ownership of this buffer remains with the stream, and the buffer
  //   remains valid only until some other method of the stream is called
  //   or the stream is destroyed.
  // * It is legal for the returned buffer to have zero size, as long
  //   as repeatedly calling Next() eventually yields a buffer with non-zero
  //   size.
  virtual bool Next(const void** data, int* size) = 0;

  // Backs up a number of bytes, so that the next call to Next() returns
  // data again that was already returned by the last call to Next().  This
  // is useful when writing procedures that are only supposed to read up
  // to a certain point in the input, then return.  If Next() returns a
  // buffer that goes beyond what you wanted to read, you can use BackUp()
  // to return to the point where you intended to finish.
  //
  // Preconditions:
  // * The last method called must have been Next().
  // * count must be less than or equal to the size of the last buffer
  //   returned by Next().
  //
  // Postconditions:
  // * The last "count" bytes of the last buffer returned by Next() will be
  //   pushed back into the stream.  Subsequent calls to Next() will return
  //   the same data again before producing new data.
  virtual void BackUp(int count) = 0;

  // Skips a number of bytes.  Returns false if the end of the stream is
  // reached or some input error occurred.  In the end-of-stream case, the
  // stream is advanced to the end of the stream (so ByteCount() will return
  // the total size of the stream).
  virtual bool Skip(int count) = 0;

  // Returns the total number of bytes read since this object was created.
  virtual int64 ByteCount() const = 0;


 private:
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(ZeroCopyInputStream);
};

// Abstract interface similar to an output stream but designed to minimize
// copying.
class LIBPROTOBUF_EXPORT ZeroCopyOutputStream {
 public:
  inline ZeroCopyOutputStream() {}
  virtual ~ZeroCopyOutputStream();

  // Obtains a buffer into which data can be written.  Any data written
  // into this buffer will eventually (maybe instantly, maybe later on)
  // be written to the output.
  //
  // Preconditions:
  // * "size" and "data" are not NULL.
  //
  // Postconditions:
  // * If the returned value is false, an error occurred.  All errors are
  //   permanent.
  // * Otherwise, "size" points to the actual number of bytes in the buffer
  //   and "data" points to the buffer.
  // * Ownership of this buffer remains with the stream, and the buffer
  //   remains valid only until some other method of the stream is called
  //   or the stream is destroyed.
  // * Any data which the caller stores in this buffer will eventually be
  //   written to the output (unless BackUp() is called).
  // * It is legal for the returned buffer to have zero size, as long
  //   as repeatedly calling Next() eventually yields a buffer with non-zero
  //   size.
  virtual bool Next(void** data, int* size) = 0;

  // Backs up a number of bytes, so that the end of the last buffer returned
  // by Next() is not actually written.  This is needed when you finish
  // writing all the data you want to write, but the last buffer was bigger
  // than you needed.  You don't want to write a bunch of garbage after the
  // end of your data, so you use BackUp() to back up.
  //
  // Preconditions:
  // * The last method called must have been Next().
  // * count must be less than or equal to the size of the last buffer
  //   returned by Next().
  // * The caller must not have written anything to the last "count" bytes
  //   of that buffer.
  //
  // Postconditions:
  // * The last "count" bytes of the last buffer returned by Next() will be
  //   ignored.
  virtual void BackUp(int count) = 0;

  // Returns the total number of bytes written since this object was created.
  virtual int64 ByteCount() const = 0;

  // Write a given chunk of data to the output.  Some output streams may
  // implement this in a way that avoids copying. Check AllowsAliasing() before
  // calling WriteAliasedRaw(). It will GOOGLE_CHECK fail if WriteAliasedRaw() is
  // called on a stream that does not allow aliasing.
  //
  // NOTE: It is caller's responsibility to ensure that the chunk of memory
  // remains live until all of the data has been consumed from the stream.
  virtual bool WriteAliasedRaw(const void* data, int size);
  virtual bool AllowsAliasing() const { return false; }


 private:
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(ZeroCopyOutputStream);
};

```