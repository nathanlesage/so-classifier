# This is the annotation module that performs the automatic gender annotation
# based on a few heuristics.
import tqdm
from . import parser
import os
import shutil

def safe_clear_window():
  """
  Prepares the terminal to be written to using print_window without overwriting
  any preexisting contents (basically blank out by printing a bunch of newlines)
  """
  _, lines = shutil.get_terminal_size()
  for idx in range(lines):
    print("\033[2K") # 2K = Clear entire line

def print_window(contents):
  """
  Writes the given contents to the terminal window, filling out the full
  contents. This command will overwrite all lines in the terminal window, so you
  should make sure it will only overwrite safe contents.
  """
  _, lines = shutil.get_terminal_size()
  print(f"\033[{lines}A", end="") # Navigate to the top-left corner of the window
  # Overwrite everything
  safe_clear_window()
  # ... and go back up
  print(f"\033[{lines}A", end="")

  # Print out each line, overwriting anything until the end of the line.
  for line in contents:
    print(f"{line}\033[K", end="")

def annotate (reverse):
  """
  Facilitates the data annotation process using the labels given by labels.ini
  """

  labels = []
  with open("labels.ini", "r") as fp:
    labels = fp.read().splitlines()

  # Clear out the window before starting the annotation process
  safe_clear_window()

  for dirpath, dirnames, filenames in os.walk("data_processed"):
    for filename in sorted(filenames, reverse=reverse):
      if not filename.endswith('.txt'):
        # Especially macOS and Windows are pretty good at throwing files such as
        # thumbs.db or .DS_STORE into your directories without asking you first.
        continue

      if filename != '34430.txt' and filename != '36035.txt':
        continue # DEBUG

      # Make sure we don't add a file that's already present in the annotated
      # directory.
      if os.path.isfile("data_annotated/" + filename):
        continue

      # Retrieve all sentences
      input_file = open("data_processed/" + filename, 'r', encoding='utf-8')
      sentences = list(input_file)
      input_file.close()

      write_stream = open("data_annotated/" + filename, 'w', encoding='utf-8')
      for idx, sentence in enumerate(sentences):
        # Ask the user to provide labels for the sentences.
        title = f"File: {filename}, sentence {idx + 1} of {len(sentences)}"
        prev_sen = "<none>"
        if idx > 0:
          prev_sen = sentences[idx - 1].strip()
        next_sen = "<none>"
        if idx < len(sentences) - 1:
          next_sen = sentences[idx + 1].strip()

        annotated_sentence = {
          'text': sentence.strip(), # Remove the trailing newlines
          'subject': labels[0],
          'object': labels[1]
        }

        try:
          annotate_coder(annotated_sentence, labels, window_title=title, prev_sen=prev_sen, next_sen=next_sen)
        except KeyboardInterrupt:
          # Clean up and return early
          write_stream.close()
          os.remove("data_annotated/" + filename)
          return

        # Finally: Write to file.
        write_stream.write(f"{annotated_sentence['text']}\t{annotated_sentence['subject']}\t{annotated_sentence['object']}\n")
      write_stream.close()

def annotate_coder(sentence, labels, window_title="", prev_sen="", next_sen=""):
  valid_pos_tags = [ 'PRON', 'NOUN', 'PROPN' ]
  # Collect a list of dependency relations which we can safely ignore
  ignore_deprels = [
    # "ccomp",
    # "acl:relcl",
    "vocative",
    "expl",
    "discourse",
    # WHAT? Why are we ignoring flat and compounds here? Easy: So we ONLY
    # assign the head of the MWE a gender, and then, after the user is done,
    # we will have a second run of our algorithm, which can then make use of
    # the newly added gender to assign it throughout the compound.
    "flat",
    "compound",
    "appos"
  ]

  subject_label = None
  object_label = None

  labels_str = []
  for idx, label in enumerate(labels):
    labels_str.append(f"{idx + 1}={label}")
  labels_str = "(" + "; ".join(labels_str) + ")"

  # First ask for the subject
  print_window([
    window_title + "\n",
    "=" * len(window_title) + "\n\n",
    "PREV: " + prev_sen + "\n\n\n\n",
    "      " + sentence['text'] + "\n\n\n\n",
    "NEXT: " + next_sen + "\n\n",
    f'Label the subject {labels_str}: '
  ])

  # Ask the user for an input
  user_input = input()
  index = int(user_input, base=10)
  if index > 0 and index <= len(labels):
    sentence['subject'] = labels[index - 1]

  # Second ask for the object
  print_window([
    window_title + "\n",
    "=" * len(window_title) + "\n\n",
    "PREV: " + prev_sen + "\n\n\n\n",
    "      " + sentence['text'] + "\n\n\n\n",
    "NEXT: " + next_sen + "\n\n",
    f'Label the object {labels_str}: '
  ])

  # Ask the user for an input
  user_input = input()
  index = int(user_input, base=10)
  if index > 0 and index <= len(labels):
    sentence['object'] = labels[index - 1]
