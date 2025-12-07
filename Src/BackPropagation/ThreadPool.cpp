#include "ThreadPool.h"
#include "DebugLib.h"

ThreadPool::ThreadPool (
  size_t  NumOfThreads
  )
{
  Stop = false;

  //
  // Creating worker threads
  //
  for (size_t Idx = 0; Idx < NumOfThreads; ++Idx) {
    Threads.emplace_back([this] {
      while (true) {
        function<void()> task;
        // The reason for putting the below code
        // here is to unlock the queue before
        // executing the task so that other
        // threads can perform enqueue tasks
        {
          // Locking the queue so that data
          // can be shared safely
          unique_lock<mutex> lock(QueueMutex);

          // Waiting until there is a task to
          // execute or the pool is stopped
          QueueStateCV.wait(lock, [this] {
            return (!Tasks.empty() || Stop);
          });

          // exit the thread in case the pool
          // is stopped and there are no tasks
          if (Stop && Tasks.empty()) {
            return;
          }

          // Get the next task from the queue
          task = move(Tasks.front());
          Tasks.pop();
        }

        task();

        {
          // Notify potentially waiting threads
          // that a task has been completed
          unique_lock<mutex> lock(QueueMutex);
          if (Tasks.empty()) {
            AllTasksDoneCV.notify_all();
          }
        }
      }
    });
  }
}

ThreadPool::~ThreadPool (
  void
  )
{
  {
    // Lock the queue to update the stop flag safely
    unique_lock<mutex> lock(QueueMutex);
    Stop = true;
  }

  // Notify all threads
  QueueStateCV.notify_all();

  // Joining all worker threads to ensure they have
  // completed their tasks
  for (auto& thread : Threads) {
    thread.join();
  }
}

void
ThreadPool::Enqueue (
  function<void ()>  Task
  )
{
  {
    unique_lock<std::mutex> lock(QueueMutex);
    Tasks.emplace(move(Task));
  }
  QueueStateCV.notify_one();
}

unsigned int
ThreadPool::GetNumOfThreads (
  void
  )
{
  return (unsigned int)Threads.size();
}

void
ThreadPool::WaitForAllTasksDone(
  void
  )
{
  // 1. Lock the queue mutex
  std::unique_lock<std::mutex> lock(QueueMutex);

  // 2. Wait until the task queue is empty.
  // We rely on the worker threads to keep notifying us when they take a task
  // until the queue is finally empty.
  AllTasksDoneCV.wait(lock, [this] {
    return Tasks.empty();
  });

  // When tasks_.empty() is true, the function returns.
  // The lock is released automatically.
}