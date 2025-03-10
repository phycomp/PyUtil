tornado.ioloop — Main event loop
An I/O event loop for non-blocking sockets.

In Tornado 6.0, IOLoop is a wrapper around the asyncio event loop, with a slightly different interface. The IOLoop interface is now provided primarily for backwards compatibility; new code should generally use the asyncio event loop interface directly. The IOLoop.current class method provides the IOLoop instance corresponding to the running asyncio event loop.

IOLoop objects
classtornado.ioloop.IOLoop(*args: Any, **kwargs: Any)[source]
An I/O event loop.

As of Tornado 6.0, IOLoop is a wrapper around the asyncio event loop.

Example usage for a simple TCP server:

import asyncio
import errno
import functools
import socket

import tornado
from tornado.iostream import IOStream

async def handle_connection(connection, address):
    stream = IOStream(connection)
    message = await stream.read_until_close()
    print("message from client:", message.decode().strip())

def connection_ready(sock, fd, events):
    while True:
        try:
            connection, address = sock.accept()
        except BlockingIOError:
            return
        connection.setblocking(0)
        io_loop = tornado.ioloop.IOLoop.current()
        io_loop.spawn_callback(handle_connection, connection, address)

async def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setblocking(0)
    sock.bind(("", 8888))
    sock.listen(128)

    io_loop = tornado.ioloop.IOLoop.current()
    callback = functools.partial(connection_ready, sock)
    io_loop.add_handler(sock.fileno(), callback, io_loop.READ)
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
Most applications should not attempt to construct an IOLoop directly, and instead initialize the asyncio event loop and use IOLoop.current(). In some cases, such as in test frameworks when initializing an IOLoop to be run in a secondary thread, it may be appropriate to construct an IOLoop with IOLoop(make_current=False).

In general, an IOLoop cannot survive a fork or be shared across processes in any way. When multiple processes are being used, each process should create its own IOLoop, which also implies that any objects which depend on the IOLoop (such as AsyncHTTPClient) must also be created in the child processes. As a guideline, anything that starts processes (including the tornado.process and multiprocessing modules) should do so as early as possible, ideally the first thing the application does after loading its configuration, and before any calls to IOLoop.start or asyncio.run.

Changed in version 4.2: Added the make_current keyword argument to the IOLoop constructor.

Changed in version 5.0: Uses the asyncio event loop by default. The IOLoop.configure method cannot be used on Python 3 except to redundantly specify the asyncio event loop.

Changed in version 6.3: make_current=True is now the default when creating an IOLoop - previously the default was to make the event loop current if there wasn’t already a current one.

Running an IOLoop
staticIOLoop.current()→ IOLoop[source]
staticIOLoop.current(instance: bool = True)→ Optional[IOLoop]
Returns the current thread’s IOLoop.

If an IOLoop is currently running or has been marked as current by make_current, returns that instance. If there is no current IOLoop and instance is true, creates one.

Changed in version 4.1: Added instance argument to control the fallback to IOLoop.instance().

Changed in version 5.0: On Python 3, control of the current IOLoop is delegated to asyncio, with this and other methods as pass-through accessors. The instance argument now controls whether an IOLoop is created automatically when there is none, instead of whether we fall back to IOLoop.instance() (which is now an alias for this method). instance=False is deprecated, since even if we do not create an IOLoop, this method may initialize the asyncio loop.

Deprecated since version 6.2: It is deprecated to call IOLoop.current() when no asyncio event loop is running.

IOLoop.make_current()→ None[source]
Makes this the IOLoop for the current thread.

An IOLoop automatically becomes current for its thread when it is started, but it is sometimes useful to call make_current explicitly before starting the IOLoop, so that code run at startup time can find the right instance.

Changed in version 4.1: An IOLoop created while there is no current IOLoop will automatically become current.

Changed in version 5.0: This method also sets the current asyncio event loop.

Deprecated since version 6.2: Setting and clearing the current event loop through Tornado is deprecated. Use asyncio.set_event_loop instead if you need this.

staticIOLoop.clear_current()→ None[source]
Clears the IOLoop for the current thread.

Intended primarily for use by test frameworks in between tests.

Changed in version 5.0: This method also clears the current asyncio event loop.

Deprecated since version 6.2.

IOLoop.start()→ None[source]
Starts the I/O loop.

The loop will run until one of the callbacks calls stop(), which will make the loop stop after the current event iteration completes.

IOLoop.stop()→ None[source]
Stop the I/O loop.

If the event loop is not currently running, the next call to start() will return immediately.

Note that even after stop has been called, the IOLoop is not completely stopped until IOLoop.start has also returned. Some work that was scheduled before the call to stop may still be run before the IOLoop shuts down.

IOLoop.run_sync(func: Callable, timeout: Optional[float] = None)→ Any[source]
Starts the IOLoop, runs the given function, and stops the loop.

The function must return either an awaitable object or None. If the function returns an awaitable object, the IOLoop will run until the awaitable is resolved (and run_sync() will return the awaitable’s result). If it raises an exception, the IOLoop will stop and the exception will be re-raised to the caller.

The keyword-only argument timeout may be used to set a maximum duration for the function. If the timeout expires, a asyncio.TimeoutError is raised.

This method is useful to allow asynchronous calls in a main() function:

async def main():
    # do stuff...

if __name__ == '__main__':
    IOLoop.current().run_sync(main)
Changed in version 4.3: Returning a non-None, non-awaitable value is now an error.

Changed in version 5.0: If a timeout occurs, the func coroutine will be cancelled.

Changed in version 6.2: tornado.util.TimeoutError is now an alias to asyncio.TimeoutError.

IOLoop.close(all_fds: bool = False)→ None[source]
Closes the IOLoop, freeing any resources used.

If all_fds is true, all file descriptors registered on the IOLoop will be closed (not just the ones created by the IOLoop itself).

Many applications will only use a single IOLoop that runs for the entire lifetime of the process. In that case closing the IOLoop is not necessary since everything will be cleaned up when the process exits. IOLoop.close is provided mainly for scenarios such as unit tests, which create and destroy a large number of IOLoops.

An IOLoop must be completely stopped before it can be closed. This means that IOLoop.stop() must be called and IOLoop.start() must be allowed to return before attempting to call IOLoop.close(). Therefore the call to close will usually appear just after the call to start rather than near the call to stop.

Changed in version 3.1: If the IOLoop implementation supports non-integer objects for “file descriptors”, those objects will have their close method when all_fds is true.

staticIOLoop.instance()→ IOLoop[source]
Deprecated alias for IOLoop.current().

Changed in version 5.0: Previously, this method returned a global singleton IOLoop, in contrast with the per-thread IOLoop returned by current(). In nearly all cases the two were the same (when they differed, it was generally used from non-Tornado threads to communicate back to the main thread’s IOLoop). This distinction is not present in asyncio, so in order to facilitate integration with that package instance() was changed to be an alias to current(). Applications using the cross-thread communications aspect of instance() should instead set their own global variable to point to the IOLoop they want to use.

Deprecated since version 5.0.

IOLoop.install()→ None[source]
Deprecated alias for make_current().

Changed in version 5.0: Previously, this method would set this IOLoop as the global singleton used by IOLoop.instance(). Now that instance() is an alias for current(), install() is an alias for make_current().

Deprecated since version 5.0.

staticIOLoop.clear_instance()→ None[source]
Deprecated alias for clear_current().

Changed in version 5.0: Previously, this method would clear the IOLoop used as the global singleton by IOLoop.instance(). Now that instance() is an alias for current(), clear_instance() is an alias for clear_current().

Deprecated since version 5.0.

I/O events
IOLoop.add_handler(fd: int, handler: Callable[[int, int], None], events: int)→ None[source]
IOLoop.add_handler(fd: _S, handler: Callable[[_S, int], None], events: int)→ None
Registers the given handler to receive the given events for fd.

The fd argument may either be an integer file descriptor or a file-like object with a fileno() and close() method.

The events argument is a bitwise or of the constants IOLoop.READ, IOLoop.WRITE, and IOLoop.ERROR.

When an event occurs, handler(fd, events) will be run.

Changed in version 4.0: Added the ability to pass file-like objects in addition to raw file descriptors.

IOLoop.update_handler(fd: Union[int, _Selectable], events: int)→ None[source]
Changes the events we listen for fd.

Changed in version 4.0: Added the ability to pass file-like objects in addition to raw file descriptors.

IOLoop.remove_handler(fd: Union[int, _Selectable])→ None[source]
Stop listening for events on fd.

Changed in version 4.0: Added the ability to pass file-like objects in addition to raw file descriptors.

Callbacks and timeouts
IOLoop.add_callback(callback: Callable, *args: Any, **kwargs: Any)→ None[source]
Calls the given callback on the next I/O loop iteration.

It is safe to call this method from any thread at any time, except from a signal handler. Note that this is the only method in IOLoop that makes this thread-safety guarantee; all other interaction with the IOLoop must be done from that IOLoop’s thread. add_callback() may be used to transfer control from other threads to the IOLoop’s thread.

IOLoop.add_callback_from_signal(callback: Callable, *args: Any, **kwargs: Any)→ None[source]
Calls the given callback on the next I/O loop iteration.

Intended to be afe for use from a Python signal handler; should not be used otherwise.

Deprecated since version 6.4: Use asyncio.AbstractEventLoop.add_signal_handler instead. This method is suspected to have been broken since Tornado 5.0 and will be removed in version 7.0.

IOLoop.add_future(future: Union[Future[_T], concurrent.futures.Future[_T]], callback: Callable[[Future[_T]], None])→ None[source]
Schedules a callback on the IOLoop when the given Future is finished.

The callback is invoked with one argument, the Future.

This method only accepts Future objects and not other awaitables (unlike most of Tornado where the two are interchangeable).

IOLoop.add_timeout(deadline: Union[float, timedelta], callback: Callable, *args: Any, **kwargs: Any)→ object[source]
Runs the callback at the time deadline from the I/O loop.

Returns an opaque handle that may be passed to remove_timeout to cancel.

deadline may be a number denoting a time (on the same scale as IOLoop.time, normally time.time), or a datetime.timedelta object for a deadline relative to the current time. Since Tornado 4.0, call_later is a more convenient alternative for the relative case since it does not require a timedelta object.

Note that it is not safe to call add_timeout from other threads. Instead, you must use add_callback to transfer control to the IOLoop’s thread, and then call add_timeout from there.

Subclasses of IOLoop must implement either add_timeout or call_at; the default implementations of each will call the other. call_at is usually easier to implement, but subclasses that wish to maintain compatibility with Tornado versions prior to 4.0 must use add_timeout instead.

Changed in version 4.0: Now passes through *args and **kwargs to the callback.

IOLoop.call_at(when: float, callback: Callable, *args: Any, **kwargs: Any)→ object[source]
Runs the callback at the absolute time designated by when.

when must be a number using the same reference point as IOLoop.time.

Returns an opaque handle that may be passed to remove_timeout to cancel. Note that unlike the asyncio method of the same name, the returned object does not have a cancel() method.

See add_timeout for comments on thread-safety and subclassing.

New in version 4.0.

IOLoop.call_later(delay: float, callback: Callable, *args: Any, **kwargs: Any)→ object[source]
Runs the callback after delay seconds have passed.

Returns an opaque handle that may be passed to remove_timeout to cancel. Note that unlike the asyncio method of the same name, the returned object does not have a cancel() method.

See add_timeout for comments on thread-safety and subclassing.

New in version 4.0.

IOLoop.remove_timeout(timeout: object)→ None[source]
Cancels a pending timeout.

The argument is a handle as returned by add_timeout. It is safe to call remove_timeout even if the callback has already been run.

IOLoop.spawn_callback(callback: Callable, *args: Any, **kwargs: Any)→ None[source]
Calls the given callback on the next IOLoop iteration.

As of Tornado 6.0, this method is equivalent to add_callback.

New in version 4.0.

IOLoop.run_in_executor(executor: Optional[Executor], func: Callable[[...], _T], *args: Any)→ Future[_T][source]
Runs a function in a concurrent.futures.Executor. If executor is None, the IO loop’s default executor will be used.

Use functools.partial to pass keyword arguments to func.

New in version 5.0.

IOLoop.set_default_executor(executor: Executor)→ None[source]
Sets the default executor to use with run_in_executor().

New in version 5.0.

IOLoop.time()→ float[source]
Returns the current time according to the IOLoop’s clock.

The return value is a floating-point number relative to an unspecified time in the past.

Historically, the IOLoop could be customized to use e.g. time.monotonic instead of time.time, but this is not currently supported and so this method is equivalent to time.time.

classtornado.ioloop.PeriodicCallback(callback: Callable[[], Optional[Awaitable]], callback_time: Union[timedelta, float], jitter: float = 0)[source]
Schedules the given callback to be called periodically.

The callback is called every callback_time milliseconds when callback_time is a float. Note that the timeout is given in milliseconds, while most other time-related functions in Tornado use seconds. callback_time may alternatively be given as a datetime.timedelta object.

If jitter is specified, each callback time will be randomly selected within a window of jitter * callback_time milliseconds. Jitter can be used to reduce alignment of events with similar periods. A jitter of 0.1 means allowing a 10% variation in callback time. The window is centered on callback_time so the total number of calls within a given interval should not be significantly affected by adding jitter.

If the callback runs for longer than callback_time milliseconds, subsequent invocations will be skipped to get back on schedule.

start must be called after the PeriodicCallback is created.

Changed in version 5.0: The io_loop argument (deprecated since version 4.1) has been removed.

Changed in version 5.1: The jitter argument is added.

Changed in version 6.2: If the callback argument is a coroutine, and a callback runs for longer than callback_time, subsequent invocations will be skipped. Previously this was only true for regular functions, not coroutines, which were “fire-and-forget” for PeriodicCallback.

The callback_time argument now accepts datetime.timedelta objects, in addition to the previous numeric milliseconds.

start()→ None[source]
Starts the timer.

stop()→ None[source]
Stops the timer.

is_running()→ bool[source]
Returns True if this PeriodicCallback has been started.

New in version 4.1.
