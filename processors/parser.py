# This is the parser module that is able to parse annotated data files.

def parse_preprocessed (filename, yield_eof=False):
  """
  Parses a preprocessed (not raw!) file into a Python data structure. The result
  is an iterator which yields sentences, which contain tokens in dictionary form
  providing the following keys (see documentation of preprocess for a more
  verbose documentation):

  * text
  * lemma
  * head
  * upos
  * deprel
  * ner
  * gender [this will only be included if there are gender annotations]

  An additional flag "yield_eof" can be set to indicate the end of a file (if
  that is wanted). If that flag is set to True, this means that the generator
  will from time to time NOT yield a list, but rather a single string, indicating
  the end of a document (eof = End of File). If the flag is set to false, this
  will be suppressed, simply yielding the sentence and continuing.
  """
  with open(filename, 'r', encoding='utf-8') as file:
    sentence = []
    for idx, line in enumerate(file):
      sanitized = line.strip()
      # Also check the length of a sentence to make sure we don't have a double
      # newline, which happens at the end of a file.
      if sanitized == '' and len(sentence) > 0:
        # Empty line means end of sentence
        yield sentence
        sentence = []
        continue

      if sanitized == '':
        # Occasionally, there may be multiple newlines in the file.
        continue

      if sanitized == '<eof>':
        # An <eof>-token will ONLY be encountered in the split data and indicates
        # the end of the document. The training loop will interpret this as a
        # signal to reset the retained labels so that they are not used by the
        # classifier anymore. This means: Yield whatever we have in the sentence,
        # reset the sentence, and immediately yield an <eof>-token so the training
        # loop knows what's going on.
        if len(sentence) > 0: # Check sentence length to skip over accidental newlines in the training data
          yield sentence
          sentence = []
        if yield_eof:
          yield '<eof>'
        continue

      # Extract the information
      columns = sanitized.split("\t")
      if len(columns) == 9:
        token_id, text, lemma, head, upos, deprel, ner, subject_label, object_label = columns
        sentence.append({
          'id': int(token_id),
          'text': text,
          'lemma': lemma,
          'head': int(head),
          'upos': upos,
          'deprel': deprel,
          'ner': ner,
          'subject': subject_label,
          'object': object_label
        })
      else:
        raise ValueError(f"Wrong number of columns provided on line {idx + 1} (expected 7 or 8 columns, got {len(columns)})")

    if len(sentence) > 0:
      yield sentence # One final yield
