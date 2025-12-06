#ifndef _THREAD_POOL_H_
#define _THREAD_POOL_H_

#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
using namespace std;

// Class that represents a simple thread pool
class ThreadPool {
public:
  // // Constructor to creates a thread pool with given
  // number of threads
  ThreadPool(size_t NumOfThreads = thread::hardware_concurrency());

  // Destructor to stop the thread pool
  ~ThreadPool();

  // Enqueue task for execution by the thread pool
  void Enqueue(function<void()> Task);

  // Get number of threads of this thread pool
  unsigned int
  GetNumOfThreads (
    void
    );

  void
  WaitForAllTasksDone(
    void
    );

private:
  // Vector to store worker threads
  vector<thread> Threads;

  // Queue of tasks
  queue<function<void()> > Tasks;

  // Mutex to synchronize access to shared data
  mutex QueueMutex;

  // Condition variable to signal changes in the state of
  // the tasks queue
  condition_variable QueueStateCV;

  // Condition variable to signal threads that is waiting for the task queue being empty.
  condition_variable AllTasksDoneCV;

  // Flag to indicate whether the thread pool should stop
  // or not
  bool Stop;
};

#endif