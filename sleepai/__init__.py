"""Sleep-learning AI: a local model that learns from experience during sleep.

Vocabulary (used exactly this way throughout the code):
  wake             - the model is in use; episodes are logged
  episode          - raw record of one thing that happened during wake
  memory           - curated, keyed, sourced note formed from episodes
  sleep            - offline learning run while the machine is idle
  memory formation - sleep stage 1: episodes -> memories
  consolidation    - sleep stage 2: verified memories -> weight updates
  replay           - older memories mixed into consolidation
  dreaming         - sleep stage 3: checked self-practice
  dream report     - summary of a sleep, waiting for morning review
  morning review   - a human approves or rejects a dream report
  probe            - question testing whether a memory lives in the weights
"""
__version__ = "0.1.0"
