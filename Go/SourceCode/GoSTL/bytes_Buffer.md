# § bytes.Buffer

## 1 路径

* bytes/buffer.go

## 2 功能

* 字节buffer

## 3 基础部分

### 3.1 struct定义

buf是数据存储的地方，数据在`buf[off : len(buff)]`；off是offset的意思，读时从`buf[off]`开始，写时从`buf[len(buf)]`（即buf的最后）开始。

```go
// A Buffer is a variable-sized buffer of bytes with Read and Write methods.
// The zero value for Buffer is an empty buffer ready to use.
type Buffer struct {
	buf      []byte // contents are the bytes buf[off : len(buf)]
	off      int    // read at &buf[off], write at &buf[len(buf)]
	lastRead readOp // last read operation, so that Unread* can work correctly.
}
```

相关变量：

```go
// The readOp constants describe the last action performed on
// the buffer, so that UnreadRune and UnreadByte can check for
// invalid usage. opReadRuneX constants are chosen such that
// converted to int they correspond to the rune size that was read.
type readOp int8

// Don't use iota for these, as the values need to correspond with the
// names and comments, which is easier to see when being explicit.
const (
	opRead      readOp = -1 // Any other read operation.
	opInvalid   readOp = 0  // Non-read operation.
	opReadRune1 readOp = 1  // Read rune of size 1.
	opReadRune2 readOp = 2  // Read rune of size 2.
	opReadRune3 readOp = 3  // Read rune of size 3.
	opReadRune4 readOp = 4  // Read rune of size 4.
)
```

### 3.2 Reset

```go
// Reset resets the buffer to be empty,
// but it retains the underlying storage for use by future writes.
// Reset is the same as Truncate(0).
func (b *Buffer) Reset() {
	b.buf = b.buf[:0]
	b.off = 0
	b.lastRead = opInvalid
}
```

### 3.3 Len、Cap

注意这里的求长度的方法是`len(b.buf) - b.off`，是指buffer未被read的部分。

```go
// Len returns the number of bytes of the unread portion of the buffer;
// b.Len() == len(b.Bytes()).
func (b *Buffer) Len() int { return len(b.buf) - b.off }

// Cap returns the capacity of the buffer's underlying byte slice, that is, the
// total space allocated for the buffer's data.
func (b *Buffer) Cap() int { return cap(b.buf) }
```

## 4 写部分

### 4.1 Write、WriteString、WriteByte

* Write方法将[]byte写入Buffer中，先调用tryGrowByReslice来试图通过Reslice来获得len(p)的空间：
  * 如果成功的话，则将p复制（调用copy）过去；
  * 如果不成功的话，则调用grow来获得len(p)的空间，最后依然将p复制（调用copy）过去。

* WriteString与WriteByte基本一致，不过输入参数是string类型。

* WriteByte方法将byte写入Buffer中，先调用tryGrowByReslice来试图通过Reslice来获得1个byte的空间
  * 如果成功的话，则将其复制（实际上是直接赋值）；
  * 如果不成功的话，调用grow来获得1个byte的空间，最后依然将其复制（实际上是直接赋值）过去。

```go
// Write appends the contents of p to the buffer, growing the buffer as
// needed. The return value n is the length of p; err is always nil. If the
// buffer becomes too large, Write will panic with ErrTooLarge.
func (b *Buffer) Write(p []byte) (n int, err error) {
	b.lastRead = opInvalid
	m, ok := b.tryGrowByReslice(len(p))
	if !ok {
		m = b.grow(len(p))
	}
	return copy(b.buf[m:], p), nil
}

// WriteString appends the contents of s to the buffer, growing the buffer as
// needed. The return value n is the length of s; err is always nil. If the
// buffer becomes too large, WriteString will panic with ErrTooLarge.
func (b *Buffer) WriteString(s string) (n int, err error) {
	b.lastRead = opInvalid
	m, ok := b.tryGrowByReslice(len(s))
	if !ok {
		m = b.grow(len(s))
	}
	return copy(b.buf[m:], s), nil
}

// WriteByte appends the byte c to the buffer, growing the buffer as needed.
// The returned error is always nil, but is included to match bufio.Writer's
// WriteByte. If the buffer becomes too large, WriteByte will panic with
// ErrTooLarge.
func (b *Buffer) WriteByte(c byte) error {
	b.lastRead = opInvalid
	m, ok := b.tryGrowByReslice(1)
	if !ok {
		m = b.grow(1)
	}
	b.buf[m] = c
	return nil
}
```

### 4.2 Grow、grow、tryGrowByReslice

tryGrowByReslice是当buf使用的空间小于其容量，即其中还有空间时，直接通过reslice把buf长度截取为`l+n`（长度+需要的长度）。

Grow是将Buffer*增加*（分配）内存空间的方法。

grow是Grow具体实现。以下以3个例子讲述一下grow函数的过程。

* 场景一：第一次调用Write类方法向Buffer写入数据，并且字符小于64个byte，举例n = 1。
  * m = 0
  * 调用tryGrowByReslice，实际上无空间可Reslice，返回的第二个参数为false。
  * b.buf被分配了空间，len = 1，cap = 64，并将0（buf的起始位置）返回。
* 场景二：第一次调用Write类方法向Buffer写入数据，并且字符小于64个byte，举例n = 128。
  * m = 0
  * 调用tryGrowByReslice，实际上无空间可Reslice，返回的第二个参数为false。
  * c = 0
  * 调用makeSlice，长度：2*c +。n，即128。
  * 将b.buf调用copy复制到新分配的的区域，并将b.buf指向buf。
  * b.off置为0，bu.buf = b.buf[:m + n]
* 场景三：第n次调用Grow扩1个byte的空间，但原来cap为128，off为65，Len为10，len为75，n为54。
  * m = 10
  * 调用tryGrowByReslice，因为n = 6=54 > cap - len = 53，第二个参数返回false。
  * c = 128
  * 因为n = 54 <= c/2-m = 54，则将b.buf[b.off:]上移至b.buf，调用copy函数

```go
// tryGrowByReslice is a inlineable version of grow for the fast-case where the
// internal buffer only needs to be resliced.
// It returns the index where bytes should be written and whether it succeeded.
func (b *Buffer) tryGrowByReslice(n int) (int, bool) {
	if l := len(b.buf); n <= cap(b.buf)-l {
		b.buf = b.buf[:l+n]
		return l, true
	}
	return 0, false
}

// Grow grows the buffer's capacity, if necessary, to guarantee space for
// another n bytes. After Grow(n), at least n bytes can be written to the
// buffer without another allocation.
// If n is negative, Grow will panic.
// If the buffer can't grow it will panic with ErrTooLarge.
func (b *Buffer) Grow(n int) {
	if n < 0 {
		panic("bytes.Buffer.Grow: negative count")
	}
	m := b.grow(n)
	b.buf = b.buf[:m]
}

// smallBufferSize is an initial allocation minimal capacity.
const smallBufferSize = 64

// grow grows the buffer to guarantee space for n more bytes.
// It returns the index where bytes should be written.
// If the buffer can't grow it will panic with ErrTooLarge.
func (b *Buffer) grow(n int) int {
	m := b.Len()
	// If buffer is empty, reset to recover space.
	if m == 0 && b.off != 0 {
		b.Reset()
	}
	// Try to grow by means of a reslice.
	if i, ok := b.tryGrowByReslice(n); ok {
		return i
	}
	if b.buf == nil && n <= smallBufferSize {
		b.buf = make([]byte, n, smallBufferSize)
		return 0
	}
	c := cap(b.buf)
	if n <= c/2-m {
		// We can slide things down instead of allocating a new
		// slice. We only need m+n <= c to slide, but
		// we instead let capacity get twice as large so we
		// don't spend all our time copying.
		copy(b.buf, b.buf[b.off:])
	} else if c > maxInt-c-n {
		panic(ErrTooLarge)
	} else {
		// Not enough space anywhere, we need to allocate.
		buf := makeSlice(2*c + n)
		copy(buf, b.buf[b.off:])
		b.buf = buf
	}
	// Restore b.off and len(b.buf).
	b.off = 0
	b.buf = b.buf[:m+n]
	return m
}

// makeSlice allocates a slice of size n. If the allocation fails, it panics
// with ErrTooLarge.
func makeSlice(n int) []byte {
	// If the make fails, give a known error.
	defer func() {
		if recover() != nil {
			panic(ErrTooLarge)
		}
	}()
	return make([]byte, n)
}
```

### 4.3 总结

<u>总之，在Write时，提前分配内存空间是一个比较好的选择。</u>

## 5 读部分

### 5.1 Bytes

返回违背读取的buf。

```cpp
// Bytes returns a slice of length b.Len() holding the unread portion of the buffer.
// The slice is valid for use only until the next buffer modification (that is,
// only until the next call to a method like Read, Write, Reset, or Truncate).
// The slice aliases the buffer content at least until the next buffer modification,
// so immediate changes to the slice will affect the result of future reads.
func (b *Buffer) Bytes() []byte { return b.buf[b.off:] }
```



