import stanza
import torch
import os
from .classifier import KomninosManandhar

class Predictor:
    def __init__ (self, subject_model, object_model, labels):
        self.subj = subject_model
        self.obj = object_model
        self.stanza = stanza.Pipeline('en', processors="tokenize,pos,lemma,ner,depparse", verbose=False)
        self.labels = labels

    @classmethod
    def from_file(Predictor):
        """
        Loads the pretrained subject and object classifier's weights and returns a
        Predictor that can then be used to predict sentences. The file
        model_state.torch must exist in the data directory.
        """

        # Before we can load the file, we have to make sure we even have a file
        # to begin with.
        model_file = "data/model_state.torch"
        for dirpath, dirnames, filenames in os.walk("data"):
            for fname in sorted(filenames, reverse=True):
                if fname.endswith(".torch"):
                    model_file = "data/" + fname
                    break # We use the first found file (=newest file)

        # First load the data file
        print(f"Loading model from {model_file} ...")
        state = torch.load(model_file)

        # Now the state contains several dictionary keys we need
        word_vocab = state['word_vocab']
        context_vocab = state['context_vocab']

        # We also need the labels
        labels = state['labels']

        # Recreate the models
        subject_model = KomninosManandhar(word_vocab, context_vocab, output_dim=len(labels))
        object_model = KomninosManandhar(word_vocab, context_vocab, output_dim=len(labels))

        # Now we can initialise the weights
        subject_model.load_state_dict(state['subject_model'])
        object_model.load_state_dict(state['object_model'])

        # Set the models to evaluation mode
        subject_model.eval()
        object_model.eval()

        # Create a new Predictor and return that
        return Predictor(subject_model, object_model, labels)

    def predict (self, input):
        """
        Returns a list (for each sentence detected by Stanza) containing four-
        element-tuples: (index, sentence, subject, object)
        """
        doc = self.stanza(input)
        results = [] # Prepare the results set

        # In case of multiple detected sentences by Stanza, we want to make use
        # of the previous genders, so we need to keep them. We pass 3 as our
        # initial genders because that will make the forward() method return a
        # neutral state with no priming (b/c the last output embedding, one class
        # MORE than what we would like to predict is the padding embedding).
        previous_subj = torch.LongTensor([3])
        previous_obj = torch.LongTensor([3])

        for idx, sentence in enumerate(doc.sentences):
            tokens = sentence.to_dict()
            features = self.subj.featurise(tokens)

            # Now we have to pad the features to equal length
            max_len = 0
            for feature in features:
                if len(feature) > max_len:
                    max_len = len(feature)
            
            for feature in features:
                while len(feature) < max_len:
                    feature.append(0)

            tensor = torch.LongTensor(features).unsqueeze(0)

            subj_pred = self.subj.forward(tensor, previous_subj)
            obj_pred = self.obj.forward(tensor, previous_obj)

            # Get the highest scoring indices
            subj_label_idx = torch.argmax(subj_pred)
            obj_label_idx = torch.argmax(obj_pred)

            # Append to results
            results.append((idx, sentence.text, self.labels[subj_label_idx], self.labels[obj_label_idx]))

            # Also, save the label indices for the next iteration
            previous_subj = torch.LongTensor([subj_label_idx])
            previous_obj = torch.LongTensor([obj_label_idx])

        return results

    def to_disk (self, fname="model_state"):
        """
        Saves the predictor's model states to data/model_state.torch.
        """
        print(f"Saving predictor as {fname}.torch ...")
        torch.save({
            'subject_model': self.subj.state_dict(),
            'object_model': self.obj.state_dict(),
            # We take the vocabularies from the subject model
            'word_vocab': self.subj.word_vocab,
            'context_vocab': self.subj.context_vocab,
            # Important: Since the classifiers are bound to the labels from training,
            # we have to retain these in our saved state!
            'labels': self.labels
        }, f"data/{fname}.torch")
        print("Done!")

    def max_label_length (self):
        max = 0
        for label in self.labels:
            if len(label) > max:
                max = len(label)
        return max
