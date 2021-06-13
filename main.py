# MAIN ENTRY FILE

import sys
import stanza
import processors.data as data
import processors.annotate as annotator
import processors.splitter as splitter
from datetime import datetime, timezone
from model.train import train
from model.predictor import Predictor
import time
# from model import util # DEBUG
import os
import requests
from zipfile import ZipFile
import tqdm
from math import ceil

import argparse

def download_glove ():
  """
  Downloads the 6B GloVe embeddings (approx. 1GB)
  """
  # Get the URL ...
  print("Downloading https://nlp.stanford.edu/data/glove.6B.zip ...")
  res = requests.get("https://nlp.stanford.edu/data/glove.6B.zip", stream=True)
  if res.status_code != 200:
    print("Could not download the 6B GloVe Dataset! The server responded with code " + res.status_code + ".")
    sys.exit(1)

  # ... and write it to file
  fp = open("data/glove.6B.zip", "wb")
  total_length = int(res.headers.get('content-length'))
  # Thanks again to the internet for this beautiful piece of code <3
  for chunk in tqdm.tqdm(res.iter_content(chunk_size=1024), unit="KB", total=ceil(total_length/1024) + 1):
    if chunk:
      fp.write(chunk)
      fp.flush()
  fp.close()
  print("ZIP-file downloaded! Extracting ...")
  with ZipFile("data/glove.6B.zip", "r") as zf:
    files = zf.namelist()
    print("Members in archive:")
    print("\n".join(files))

    for file in files:
      if file.endswith("glove.6B.300d.txt"):
        print("Extracting member " + file + " from archive ...")
        zf.extract(file)
        break
  
  # Remove the zip file again
  os.remove("data/glove.6B.zip")
  print("Successfully extracted GloVe embeddings (300 dimensions) to data directory.")
  print("You can now train the classifier using the GloVe embeddings.")

def run_stanza (arguments):
  """
  Runs the Stanza module
  """
  if arguments.download:
    # Download the full Stanza dataset
    result = input("ATTENTION! This will download the full English Stanza corpus (approx. 400 MB). Do you wish to continue (y/n)? ")
    if result == "y" or result == "yes":
      # For a list of processors, see https://stanfordnlp.github.io/stanza/pipeline.html#processors
      stanza.download('en', processors="tokenize,pos,lemma,ner,depparse")
    sys.exit(0) # Afterwards exit normally

def run_glove (arguments):
  """
  Runs the GloVe module
  """
  if arguments.download:
    # Download the 6B GloVe dataset
    result = input("ATTENTION! This will download approximately 1GB of data. Do you wish to continue (y/n)? ")
    if result == "y" or result == "yes":
      download_glove()
    sys.exit(0) # Afterwards exit normally

def run_data (arguments):
  """
  Runs the Data module
  """
  if arguments.define_labels:
    data.define_labels()
  elif arguments.preprocess:
    # Preprocess from data_raw --> data_preprocessed
    data.preprocess()
  elif arguments.annotate:
    # Annotate from data_preprocessed --> data_annotated
    reverse = False # DEBUG
    annotator.annotate(reverse)
  elif arguments.split:
    # Split from data_annotated --> train.txt/valid.txt
    restrict = 100 # Default: Keep 100% of all files
    splitter.train_valid(restrict_to=restrict)

def run_model (arguments):
  """
  Runs the Model module
  """
  if arguments.train is not None:
    # Train a new model, optionally with a certain number of epochs
    predictor = None
    if len(arguments.train) > 0:
      predictor = train(n_epochs=arguments.train[0])
    else:
      predictor = train()
    # Afterwards save it
    now = datetime.now(timezone.utc)
    predictor.to_disk(fname=f"model_parameters_{now.strftime('%Y%m%d%H%M%S')}")
  elif arguments.export_embeddings:
    # Load the saved predictor ...
    predictor = Predictor.from_file()
    # ... and then dump the models to disk.
    predictor.subj.export_embeddings("subject")
    predictor.obj.export_embeddings("object")
    print("Models are saved to output directory for loading with http://projector.tensorflow.org/.")
  elif arguments.console:
    # Opens a console for prediction without training
    predictor = Predictor.from_file()
    tinker(predictor)

def tinker (predictor):
  """
  Starts up a console to interact with a predictor.
  """
  col_size = predictor.max_label_length()
  while True:
    try:
      sentence = input('\nEnter a sentence to classify ("exit" to quit): ')
      if sentence == 'exit':
        print("Goodbye!")
        sys.exit(0)
      print("")
      results = predictor.predict(sentence)
      print(f"Index | {'Subject'.ljust(col_size)} | {'Object'.ljust(col_size)} | Sentence")
      print(f"------|-{'-' * col_size}-|-{'-' * col_size}-|---------")
      for result in results:
        idx, sentence, subj, obj = result
        print(f"{str(idx).rjust(5)} | {subj.ljust(col_size)} | {obj.ljust(col_size)} | {sentence}")
    except KeyboardInterrupt:
      print("Goodbye!")
      sys.exit(0)

if __name__ == "__main__":
  # We'll create sub-parsers for all commands we can invoke here
  parser = argparse.ArgumentParser(
    description="Classifies arbitrary textual data using custom labels." # ,
    # epilog="This will be printed below the help!"
  )

  # Save the module's name to the variable "module" for easy branching
  subparsers = parser.add_subparsers(dest="module")

  # Create the Stanza module
  parser_stanza = subparsers.add_parser("stanza", help="Control the Stanza pipeline")
  group_stanza = parser_stanza.add_mutually_exclusive_group(required=True)
  group_stanza.add_argument("-d", "--download", action="store_true", help="Download the required stanza model data")

  # Create the GloVe module
  parser_glove = subparsers.add_parser("glove", help="Retrieves the GloVe embeddings")
  group_glove = parser_glove.add_mutually_exclusive_group(required=True)
  group_glove.add_argument("-d", "--download", action="store_true", help="Downloads the 6B GloVe dataset for use in the model")

  # Create the Data module
  parser_data = subparsers.add_parser("data", help="Work with your raw data")
  group_data = parser_data.add_mutually_exclusive_group(required=True)
  group_data.add_argument("-d", "--define-labels", action="store_true", help="Define the labels used for your task")
  group_data.add_argument("-p", "--preprocess", action="store_true", help="Parse raw data into dependency trees")
  group_data.add_argument("-a", "--annotate", action="store_true", help="Annotate preprocessed data with labels")
  group_data.add_argument("-s", "--split", action="store_true", help="Create a train/valid split from the annotated data")

  # Create the Model module
  parser_model = subparsers.add_parser("model", help="Train a model")
  group_model = parser_model.add_mutually_exclusive_group(required=True)
  group_model.add_argument("-t", "--train", nargs='*', type=int, help="Trains a model, optionally specifying the number of epochs", metavar="epochs")
  group_model.add_argument("-e", "--export-embeddings", action="store_true", help="Exports the embeddings of a trained model ready for the TensorFlow projector")
  group_model.add_argument("-c", "--console", action="store_true", help="Start an interactive console to tinker with the trained model")

  # Always print the help message if there were no commands given.
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(0)

  # Run the corresponding module
  arguments = parser.parse_args()
  if arguments.module == 'stanza':
    run_stanza(arguments)
  elif arguments.module == 'glove':
    run_glove(arguments)
  elif arguments.module == 'data':
    run_data(arguments)
  elif arguments.module == 'model':
    run_model(arguments)
