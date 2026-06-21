Prerequisites:

you need install pytest + pyyaml to use these
pip install pytest pyyaml 

then start by
pytest check_phoneme_lengths.py -v
pytest test_monotonic_align.py -v


What it does: 
1) check_phoneme_lengths:
   checks if all your training data / phoneme strings are below the PLBERT limit of 512

2) test_monotonic_align:
   checks if the monotonic_align package works correctly