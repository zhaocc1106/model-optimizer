# Tensorrt inferer. Use as following:
# trt_inferer = trt_inferer.SimpleTrtInferer(trt_engine_path=trt_path,
#                                            max_inp_shape=(1, MAX_HEIGHT, MAX_WIDTH, 3),
#                                            max_out_shape=(1, MAX_HEIGHT, MAX_WIDTH, 3),
#                                            device_id=device_id)
# or
# trt_inferer = trt_inferer.MultiWorkersTrtInferer(trt_engine_path=trt_path,
#                                                  max_inp_shape=(1, MAX_HEIGHT, MAX_WIDTH, 3),
#                                                  max_out_shape=(1, MAX_HEIGHT, MAX_WIDTH, 3),
#                                                  workers_count=2,
#                                                  queue_max_size=100,
#                                                  device_id=device_id)
# output = trt_inferer.infer(input, timeout=10)

import os
import time

import numpy as np
import threading
import pycuda.driver as cuda
import tensorrt as trt
from queue import Queue, Empty, Full


class TrtInferer:
    """Base class for tensorrt inferer."""

    def __init__(self, trt_engine_path, max_inp_shape, max_out_shape):
        """Constructor function.

        Args:
            trt_engine_path: Tensorrt engine path.
            max_inp_shape: Max input shape.
            max_out_shape: Max output shape.
        """
        self.__trt_engine_path = trt_engine_path
        self.__max_inp_shape = max_inp_shape
        self.__max_out_shape = max_out_shape

    def _load_trt_engine(self):
        """Load tensorrt engine."""
        # Deserialize engine from binary file.
        print('Loading tensorrt engine...')
        if not os.path.exists(self.__trt_engine_path):
            print('Trt engine file not existed: {}'.format(self.__trt_engine_path))
            return None
        with open(self.__trt_engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(serialized_engine)

        # print engine layer description.
        print('engine layer_info:\n{}'.format(
            engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat(1))))
        return engine

    def _trt_malloc(self):
        """Malloc host memory and device memory for trt infer."""
        h_output = cuda.pagelocked_empty(trt.volume(self.__max_out_shape), dtype=np.float32)
        d_input = cuda.mem_alloc(trt.volume(self.__max_inp_shape) * 4)
        d_output = cuda.mem_alloc(trt.volume(self.__max_out_shape) * 4)
        return h_output, d_input, d_output

    def _trt_infer(self, inp_idx, out_idx, h_input, h_output, d_input, d_output, trt_ctx, stream):
        """Perform trt infer

        Args:
            inp_idx: Tensorrt engine input binding index.
            out_idx: Tensorrt engine output binding index.
            h_input: Input data in host mem.
            h_output: Host mem to save output.
            d_input: Device mem to save input.
            d_output: Device mem to save output.
            trt_ctx: Tensorrt context.
            stream: Cuda stream.

        Returns:
            output.
        """
        # begin_time = time.time()
        begin_event = cuda.Event()
        h2d_event = cuda.Event()
        inf_event = cuda.Event()
        d2h_event = cuda.Event()
        # set true input shape.
        trt_ctx.set_binding_shape(inp_idx, h_input.shape)
        # get true output volume.
        output_shape = trt_ctx.get_binding_shape(out_idx)
        output_volume = trt.volume(output_shape)
        # copy mem from host into device.
        begin_event.record(stream)
        cuda.memcpy_htod_async(d_input, h_input, stream)
        h2d_event.record(stream)
        # execute infer
        trt_ctx.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        inf_event.record(stream)
        # copy mem from device into host.
        cuda.memcpy_dtoh_async(h_output[:output_volume], d_output, stream)
        d2h_event.record(stream)
        # sync for stream
        stream.synchronize()
        print('h2d time: {} ms, inf time: {} ms, d2h time: {} ms, host time: {} ms.'
              .format(h2d_event.time_since(begin_event),
                      inf_event.time_since(h2d_event),
                      d2h_event.time_since(inf_event),
                      d2h_event.time_since(begin_event)))
        # print('total time: {} ms'.format((time.time() - begin_time) * 1000))
        # print(h_output)
        return np.reshape(h_output[:output_volume], output_shape)

    def _warm_up(self, cuda_ctx, inp_idx, out_idx, h_output, d_input, d_output, trt_ctx, stream):
        """Warmup trt inferer.

        Args:
            cuda_ctx: Cuda context.
            inp_idx: Tensorrt engine input binding index.
            out_idx: Tensorrt engine output binding index.
            h_input: Input data in host mem.
            h_output: Host mem to save output.
            d_input: Device mem to save input.
            d_output: Device mem to save output.
            trt_ctx: Tensorrt context.
            stream: Cuda stream.
        """
        cuda_ctx.push()
        self._trt_infer(inp_idx,
                        out_idx,
                        np.random.random(size=self.__max_inp_shape).astype('float32'),
                        h_output,
                        d_input,
                        d_output,
                        trt_ctx,
                        stream)
        cuda_ctx.pop()

    def destroy(self):
        pass

    def infer(self, input, timeout):
        """Infer.

        Args:
            input: Input.
            timeout: Timeout.

        Returns:
            Return output if successfully, or None if failed.
        """
        pass


class MultiWorkersTrtInferer(TrtInferer):
    """Multi worker threads pool tensorrt inferer."""

    class InferJob:
        def __init__(self, input):
            self.__input = input
            self.__output = None
            self.__complete_condition = threading.Condition()

        @property
        def input(self):
            return self.__input

        @property
        def output(self):
            return self.__output

        @output.setter
        def output(self, value):
            self.__output = value

        def wait(self, timeout=None):
            with self.__complete_condition:
                ret = self.__complete_condition.wait(timeout)
            if ret:
                return self.__output
            else:
                return None

        def job_done(self):
            with self.__complete_condition:
                self.__complete_condition.notify()

    def __init__(self, trt_engine_path, max_inp_shape, max_out_shape, workers_count=1, queue_max_size=100, device_id=0):
        """Constructor.
        Args:
            trt_engine_path: Tensorrt engine path.
            max_inp_shape: Maximum input shape.
            max_out_shape: Maximum output shape.
            workers_count: Workers count.
            queue_max_size: Queue max size.
            device_id: Device id.
        """
        TrtInferer.__init__(self, trt_engine_path, max_inp_shape, max_out_shape)
        self.__device_id = device_id
        self.__workers_count = workers_count
        self.__started_workers_count = 0  # Started workers count used to wait inferer start completely.
        self.__started_workers_count_lock = threading.Lock()  # Lock to protect __started_workers_count.
        self.__queues = []  # Each worker has one queue.
        self.__worker_threads = []  # All worker threads.
        self.__queue_max_size = queue_max_size  # Queue max size.
        self.__stop = False
        self.__current_submit_idx = 0  # The current worker idx to submit new infer job.
        self.__submit_idx_lock = threading.Lock()  # Lock to protect __submit_idx_lock.

        cuda.init()
        for i in range(self.__workers_count):
            self.__queues.append(Queue(self.__queue_max_size))
            new_worker_thread = threading.Thread(target=self.__worker_func, args=[i])
            self.__worker_threads.append(new_worker_thread)
            new_worker_thread.start()

        # Wait all workers start completely.
        while True:
            with self.__started_workers_count_lock:
                if self.__started_workers_count == self.__workers_count:
                    break
            time.sleep(0.1)
        print('MultiWorkersTrtInferer started completely!')

    def destroy(self):
        """Destroy inferer.(Multi-thread not safe)"""
        self.__stop = True
        for worker_thread in self.__worker_threads:
            worker_thread.join()
        self.__worker_threads.clear()
        self.__queues.clear()

    def __worker_func(self, idx):
        """The func of worker"""
        print('MultiWorkersTrtInferer-worker[{}] begin running...'.format(idx))
        worker_cuda_ctx = cuda.Device(self.__device_id).make_context()  # Create current worker cuda ctx.
        worker_stream = cuda.Stream()  # Create current worker cuda stream.
        worker_trt_engine = self._load_trt_engine()  # Create current worker trt engine.
        worker_trt_ctx = worker_trt_engine.create_execution_context()  # Create current worker tensorrt context.
        h_output, d_input, d_output = self._trt_malloc()  # Malloc host and device memory and reuse it for all infer item.
        self._warm_up(worker_cuda_ctx,
                      worker_trt_engine['input'],
                      worker_trt_engine['output'],
                      h_output,
                      d_input,
                      d_output,
                      worker_trt_ctx,
                      worker_stream)
        with self.__started_workers_count_lock:
            self.__started_workers_count += 1
        while not self.__stop:
            try:
                begin_time = time.time()
                infer_job = self.__queues[idx].get(block=True, timeout=10)
                worker_cuda_ctx.push()  # Make current worker cuda contex as active contex.
                infer_job.output = self._trt_infer(worker_trt_engine['input'],
                                                   worker_trt_engine['output'],
                                                   np.ascontiguousarray(infer_job.input),
                                                   h_output,
                                                   d_input,
                                                   d_output,
                                                   worker_trt_ctx,
                                                   worker_stream)
                worker_cuda_ctx.pop()  # Make current worker contex not active.
                infer_job.job_done()
                print('MultiWorkersTrtInferer-worker[{}] process job used time: {} ms.'
                      .format(idx, (time.time() - begin_time) * 1000))
            except Empty:
                print('MultiWorkersTrtInferer-worker[{}] no job, continue waiting...'.format(idx))
                continue
        worker_cuda_ctx.pop()
        print('MultiWorkersTrtInferer-worker[{}] stopped'.format(idx))

    def infer(self, input, timeout):
        """Infer.(Multi-thread safe)

        Args:
            input: Input.
            timeout: Timeout.

        Returns:
            Return output if successfully, or None if failed.
        """
        # Round-robin all worker queue.
        with self.__submit_idx_lock:
            submit_idx = self.__current_submit_idx
            self.__current_submit_idx += 1
            if self.__current_submit_idx == self.__workers_count:
                self.__current_submit_idx = 0

        try:
            infer_job = self.InferJob(input)
            self.__queues[submit_idx].put(infer_job, timeout=timeout)
        except Full:
            print('Queue[{}] have been full.'.format(submit_idx))
            return None
        return infer_job.wait(timeout)


class SimpleTrtInferer(TrtInferer):
    """Simple single worker tensorrt inferer"""

    def __init__(self, trt_engine_path, max_inp_shape, max_out_shape, device_id=0):
        """Constructor function.

        Args:
            trt_engine_path: Tensorrt engine path.
            max_inp_shape: Max input shape.
            max_out_shape: Max output shape.
            device_id: Device id.
        """
        TrtInferer.__init__(self, trt_engine_path, max_inp_shape, max_out_shape)
        cuda.init()
        self.__cuda_ctx = cuda.Device(device_id).make_context()  # Create cuda ctx.
        self.__stream = cuda.Stream()  # Create cuda stream.
        self.__trt_engine = self._load_trt_engine()  # Create trt engine.
        self.__trt_ctx = self.__trt_engine.create_execution_context()  # Create tensorrt context.
        self.__h_output, self.__d_input, self.__d_output = self._trt_malloc()  # Malloc host and device memory and reuse it for all infer item.
        self.__cuda_ctx.pop()
        self._warm_up(self.__cuda_ctx,
                      self.__trt_engine['input'],
                      self.__trt_engine['output'],
                      self.__h_output,
                      self.__d_input,
                      self.__d_output,
                      self.__trt_ctx,
                      self.__stream)
        self.__infer_lock = threading.Lock()  # Infer interface lock.

    def infer(self, input, timeout):
        """Infer.(Multi-thread safe)

        Args:
            input: Input.
            timeout: Timeout.

        Returns:
            Return output if successfully, or None if failed.
        """

        if not self.__infer_lock.acquire(blocking=True, timeout=timeout):
            print('Infer timeout.')
            return None

        try:
            self.__cuda_ctx.push()
            output = self._trt_infer(self.__trt_engine['input'],
                                     self.__trt_engine['output'],
                                     np.ascontiguousarray(input),
                                     self.__h_output,
                                     self.__d_input,
                                     self.__d_output,
                                     self.__trt_ctx,
                                     self.__stream)
            self.__cuda_ctx.pop()
            return output
        except Exception as err:
            print('Infer exception: {}'.format(err))
            return None
        finally:
            self.__infer_lock.release()
