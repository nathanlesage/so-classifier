import os
import random
import stanza
import tqdm

def train_valid (restrict_to=100):
  """
  Creates a train/validation split based on the annotated data. The parameter
  restrict_to can be used to arbitrarily constrain the amount of files considered
  for the split. If set, the number will be interpreted as a percentage
  (0-100) of files to keep. Files will then be randomly omitted.
  """
  print(f"Splitter started. Generating training and validation datasets (ratio 4:1, keeping {restrict_to}% of files).")
  for dirpath, dirnames, filenames in os.walk("data_annotated"):
    print(f"Found {len(filenames)} files in the annotation directory.")
    # We have to make sure we follow this pattern in creating the split:
    # Training vs. Validation --> 4:1
    valid_files = []
    train_files = []
    if len(filenames) < 5:
      print("WARNING: Less than five files -- will duplicate files instead of splitting!!!")
      valid_files += filenames
      train_files += filenames
    else:
      for idx, f in enumerate(filenames):
        if idx % 4 == 0:
          valid_files.append(f)
        else:
          train_files.append(f)

    # Now make sure to omit as many files as necessary
    keep_valid = round(len(valid_files) / 100 * restrict_to)
    keep_train = round(len(train_files) / 100 * restrict_to)

    if keep_valid < 1:
      keep_valid = 1
    if keep_train < 1:
      keep_train = 1

    # Shuffling the lists before splicing the required
    # amounts is equal to randomly dropping elements.
    random.shuffle(valid_files)
    random.shuffle(train_files)

    valid_files = valid_files[:keep_valid] # --snip--
    train_files = train_files[:keep_train] # --snip--

    print(f"We have split the data into {len(valid_files)} validation and {len(train_files)} training files.")
    print("Writing to disk ...")

    # Now write them to disk
    nlp = stanza.Pipeline('en', processors="tokenize,pos,lemma,ner,depparse", verbose=False)

    valid_data = open("data/valid.txt", 'w', encoding='utf-8')
    files_to_data(nlp, valid_data, valid_files, dirpath)
    valid_data.close()

    train_data = open("data/train.txt", 'w', encoding='utf-8')
    files_to_data(nlp, train_data, train_files, dirpath)
    train_data.close()

    valid_size = round(os.path.getsize("data/valid.txt") / 1024, 2)
    train_size = round(os.path.getsize("data/train.txt") / 1024, 2)

    print(f"Done! The training data contains {train_size} KB, the validation set contains {valid_size} KB.")

def files_to_data(nlp, out_stream, files, dirpath):
  """
  Writes a list of files to disk, parsing them to dependency trees if applicable.
  """
  for filename in tqdm.tqdm(files, desc="Processing files", unit="file"):
      with open(dirpath + "/" + filename, "r") as fp:
        contents = fp.read().strip() # Remove potential newlines at the end
        if contents == '':
          continue

        # Split into constituent lines, extract sentence, subject and object,
        # parse the sentence into a dependency tree and write that into the
        # training and validation dataset.
        lines = contents.splitlines(keepends=False)
        for line in tqdm.tqdm(lines, desc="Processing lines", unit="line"):
          sentence, subject_label, object_label = line.split("\t")
          parsed_doc = nlp(sentence)
          for sentence in parsed_doc.sentences:
            for token in sentence.to_dict():
              # text    Original input text
              # lemma   The base form of the word
              # head    The head for this word in the sentence (position, w/ pseudo-root)
              # upos    The token's POS tag (according to Universal POS)
              # deprel  The dependency relation of that word with regard to the sentence (e.g., nsubj, det, punct, or root)
              # ner     The named entity recognition (if applicable, O if not), follows this: https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)
              out_stream.write(f"{token['id']}\t{token['text']}\t{token['lemma']}\t{token['head']}\t{token['upos']}\t{token['deprel']}\t{token['ner']}\t{subject_label}\t{object_label}\n")
            out_stream.write("\n") # End sentences with an additional empty line
        out_stream.write("<eof>\n\n") # End with the <eof>-token
