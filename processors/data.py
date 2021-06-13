import os
from . import parser
from nltk.tokenize import sent_tokenize

def define_labels ():
  """
  Allows users to create labels with which their classifier works
  """
  if os.path.isfile("labels.ini"):
    choice = input("WARNING: There is a labels.ini file present. Running this command will overwrite this info. Continue? (y/n) ")
    if choice.lower() != "y" and choice.lower() != "yes":
      print("Aborting label creation ...")
      sys.exit(0)

  print("You will now be asked for labels that you can afterwards assign to your raw data.")
  print("These labels will be what the classifier's will predict for each subject and object.")
  print("Thus, make sure you add all you need. You can always manually edit labels.txt,")
  print("but then you cannot use the previous classifier anymore!")
  print("")
  # Now let the user define their labels
  labels = []
  while True:
    try:
      label = input(f"{len(labels)} labels. Enter the next label ('exit' or Ctrl+C to finish): ")
      if label.strip() == 'exit':
        break
      if label.strip() != '':
        labels.append(label.strip())
    except KeyboardInterrupt:
      # Allow users to use keyboard interrupt
      break
  fp = open("labels.ini", "w+")
  fp.write("\n".join(labels))
  fp.close()
  print("Label definition successful!")

def preprocess ():
  """
  Processes the files in the raw-directory. What this will do is turn any document
  into those documents where there is one sentence per line guaranteed.
  """
  # We have to use the same processors which we downloaded earlier in the main script.
  # Why do we need NER? Because that might help greatly in writing an algorithm
  # that produces the silver-standard data.
  for dirpath, dirnames, filenames in os.walk("data_raw"):
    for filename in sorted(filenames):
      if not filename.endswith('.txt'):
        # Especially macOS and Windows are pretty good at throwing files such as
        # thumbs.db or .DS_STORE into your directories without asking you first.
        print(f"Ignoring file {filename}.")
        continue

      # Make sure we don't add a file that's already present in the preprocessed
      # directory.
      if os.path.isfile("data_processed/" + filename):
        print(f"File {filename} is already processed -- skipping.")
        continue

      print(f"Processing input file {filename} ...")
      with open(dirpath + "/" + filename, encoding='utf-8') as file:
        write_stream = open("data_processed/" + filename, 'w+', encoding='utf-8')
        for line in file:
          line = line.replace('\r', '') # Convert Windows CLRF linefeeds to just LF
          if line.strip() == '':
            continue # Nothing to process here

          for sentence in sent_tokenize(line.strip()):
            # Make sure to remove any tabs linefeeds and tabs so that we have a
            # well-formatted sentence.
            sane_text = sentence.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            write_stream.write(f"{sentence}\n")
        write_stream.close()
