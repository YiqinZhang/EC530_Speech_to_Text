# EC530_Speech_to_Text

Build a queue system that will manage speech to text



## Phase 1:  Build Queue System

Implement a multiprocess queue system in `message_queue.py`. Split the processing `do_something` into multiple  process, and  return the finish time .

<img src= /img/multiprocess_q.png width="50%">




## Phase 2:  Speech to Text

Use DeepSpeech to convert speech to text in `speech.py`. Utilized `concurrent` module to achieve multiprocess of the speech conversion.

![ ](/img/speech.png)

