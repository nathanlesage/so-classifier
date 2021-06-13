# This module trains a classifier on the provided training data.
from .classifier import KomninosManandhar
from .predictor import Predictor
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from processors import parser
from collections import Counter
from . import util
import datetime

# These are constants we're using for the gender classification. We also define
# those in the utilities
UNKNOWN = 0
MALE = 1
FEMALE = 2

def baseline_accuracy (features, previous_labels):
    """
    We assume the baseline accuracy to just be the previous label
    """
    return previous_labels

# According to https://stackoverflow.com/a/66002960, a simple heuristic is
# apparently the best we can do for finding the primary subject/object of a
# sentence.
def training_examples (model, gold_data, labels, window_size=5, seq_len=25, batch_size=100):
    """
    Generates training examples based on gold-standard data.

    We follow the training examples by the two original papers. This means:
    We'll use their modification of skip-gram negative sampling (SGNS) using
    15 negative samples per correct one.

    One sample in our terms means one correct combination of features vs. 15
    random samples. Hence, we need to implement two different algorithms; one
    here and one in the featurize() method of the model itself. As we have full
    sentences w/ dependency relations and POS tags, all of this information is
    at our disposal for training and predicting. Here's what the work looks like:

    - featurize(): Takes a sentence, takes the features we have selected and
                   returns a list of those features to be fed into the model.

    - training_examples(): Retrieves a list of features by calling featurize()
                           and implements a search-algorithm to find the primary
                           subject and object of a gold-standard sentence.
                           This method returns a quintuplet with features,
                           subject, object and their previous iterations.
    """
    batch_features = torch.zeros((batch_size, seq_len, window_size), dtype=torch.long)
    batch_subj = torch.zeros(batch_size, dtype=torch.long)
    batch_prev_subj = torch.zeros(batch_size, dtype=torch.long)
    batch_obj = torch.zeros(batch_size, dtype=torch.long)
    batch_prev_obj = torch.zeros(batch_size, dtype=torch.long)

    # Passing 3 (one index above the classes we want to predict) will cause the
    # hidden state generator to use the non-trainable padding embeddings instead
    # of the trainable classes we want to predict. By overflowing our actual
    # classes to predict by one, has the effect of passing "None".
    previous_subj = len(labels)
    previous_obj = len(labels)

    batch_idx = 0
    for sentence in parser.parse_preprocessed(gold_data, yield_eof=True):
        if sentence == '<eof>':
            # The generator has signaled us that a document has ended,
            # so we must reset the previous flags
            previous_subj = len(labels)
            previous_obj = len(labels)
            continue

        features = model.featurise(sentence)
        # Then extract the labels
        subj, obj = labels.index(sentence[0]['subject']), labels.index(sentence[0]['object'])

        # Now append to all lists
        for token_position in range(seq_len):
            if token_position == len(features):
                break
            for feat in range(len(features[token_position])):
                batch_features[batch_idx][token_position][feat] = features[token_position][feat]

        batch_subj[batch_idx] = subj
        batch_prev_subj[batch_idx] = previous_subj
        batch_obj[batch_idx] = obj
        batch_prev_obj[batch_idx] = previous_obj

        # Make sure we're retaining the previous subject and object
        previous_subj = subj
        previous_obj = obj
        batch_idx += 1

        if batch_idx == batch_size:
            yield batch_features, batch_subj, batch_prev_subj, batch_obj, batch_prev_obj
            batch_idx = 0
            batch_features = torch.zeros((batch_size, seq_len, window_size), dtype=torch.long)
            batch_subj = torch.zeros(batch_size, dtype=torch.long)
            batch_prev_subj = torch.zeros(batch_size, dtype=torch.long)
            batch_obj = torch.zeros(batch_size, dtype=torch.long)
            batch_prev_obj = torch.zeros(batch_size, dtype=torch.long)

    if batch_idx > 0:
        yield batch_features[:batch_idx], batch_subj[:batch_idx], batch_prev_subj[:batch_idx], batch_obj[:batch_idx], batch_prev_obj[:batch_idx]

def train (n_epochs=10, batch_size=100, lr=1e-2):
    """
    Trains a KomninosManandhar classifier on the given training data and provides
    training info based on the validation dataset. These two files do not need
    to be provided if you make use of the "--train-valid-split" command to split
    an existing dataset into two files, which will be placed in the data directory.

    Upon finished training, the model's parameters will be saved into the data
    directory so that, after successful training, you may instantiate a model
    immediately without the need to train first.
    """
    print("Beginning training of a Komninos/Manandhar synchronous sentence classifier.")
    train_data = "data/train.txt"
    valid_data = "data/valid.txt"

    # Create the vocabularies
    word_vocab, contexts_vocab = util.make_vocabs(train_data)
    print("")
    print(f"Using vocabulary sizes of {len(word_vocab)} words and {len(contexts_vocab)} contexts.")

    labels = []
    with open("labels.ini", "r") as fp:
        labels = fp.read().splitlines()

    print(f"Binding classifiers to {len(labels)} labels: {', '.join(labels)}")

    # Prepare the models
    print("")
    print("Beginning instantiation of classifiers. This might take a while since the classifiers will be loading GloVe embeddings ...")
    subject_model = KomninosManandhar(word_vocab, contexts_vocab, output_dim=len(labels))
    object_model = KomninosManandhar(word_vocab, contexts_vocab, output_dim=len(labels))
    print("Classifiers instantiated.")
    print("")

    # Prepare the window size, because to speed up computation we want to train
    # fixed window sizes (this way we can employ matrix operations).
    window_size = 0
    for sentence in parser.parse_preprocessed(train_data):
        features = subject_model.featurise(sentence)
        if len(features) > window_size:
            window_size = len(features)

    window_size_valid = 0
    for sentence in parser.parse_preprocessed(valid_data):
        features = subject_model.featurise(sentence)
        if len(features) > window_size_valid:
            window_size_valid = len(features)

    print(f"Performing classification using window sizes of {window_size} (training) and {window_size_valid} (validation)")

    # Prepare the optimizers
    optimizer_subj = optim.Adam(subject_model.parameters(), lr=lr)
    optimizer_obj = optim.Adam(object_model.parameters(), lr=lr)
    epoch_losses = []
    accuracies = []
    baseline_acc = []

    print("Initialisation sequence complete. Entering training loop ...")

    progress = tqdm.tqdm(total=n_epochs, position=0)
    progress.set_description("Training")
    for epoch in range(n_epochs):
        subj_losses = []
        obj_losses = []
        running_accs = []
        running_baseline = []
        run = 0
        progress.update()
        # NOTE: We need to give the training examples a model where it can call
        # the featurise method on. It doesn't care about subject or object
        # model because these are gender-agnostic.
        for features, subj, previous_subj, obj, previous_obj in training_examples(subject_model, train_data, labels, window_size=window_size, batch_size=batch_size):
            # The model will be trained by _additionally_ feeding it the subject and
            # object of the previous sentence (if applicable).
            run += 1

            optimizer_obj.zero_grad()
            optimizer_subj.zero_grad()

            pred_subj = subject_model.forward(features, previous_subj)
            pred_obj = object_model.forward(features, previous_obj)

            loss_subj = F.cross_entropy(pred_subj, subj)
            loss_obj = F.cross_entropy(pred_obj, obj)

            subj_losses.append(loss_subj.item())
            obj_losses.append(loss_obj.item())

            loss_subj.backward()
            loss_obj.backward()

            optimizer_obj.step()
            optimizer_subj.step()

            # Start evaluation
            subject_model.eval()
            object_model.eval()
            with torch.no_grad():
                correct_subj = 0
                correct_obj = 0
                baseline_subj = 0
                baseline_obj = 0
                examples = 0
                for features, subj, previous_subj, obj, previous_obj in training_examples(subject_model, valid_data, labels, window_size=window_size_valid, batch_size=batch_size):
                    actual_batch_size = features.shape[0] # The last batch will be smaller
                    examples += actual_batch_size
                    valid_subj_pred = subject_model.forward(features, previous_subj)
                    valid_obj_pred = object_model.forward(features, previous_obj)
                    subj_pred = torch.argmax(valid_subj_pred, axis=1)
                    obj_pred = torch.argmax(valid_obj_pred, axis=1)
                    correct_subj += torch.sum(torch.eq(subj_pred, subj)).item()
                    correct_obj += torch.sum(torch.eq(obj_pred, obj)).item()

                    # Our definition of baseline will be a random permutation
                    # of the correct gender labels across this batch.
                    permutated_subj = subj[torch.randperm(actual_batch_size)]
                    permuated_obj = obj[torch.randperm(actual_batch_size)]
                    baseline_subj += torch.sum(torch.eq(subj, permutated_subj)).item()
                    baseline_obj += torch.sum(torch.eq(obj, permuated_obj)).item()

                accuracy = (correct_subj / examples + correct_obj / examples) / 2
                bl_acc = (baseline_subj / examples + baseline_obj / examples) / 2
                running_accs.append(accuracy)
                running_baseline.append(bl_acc)

            # Update the progress bar with our new information
            progress.set_postfix({
                'subj_loss': torch.mean(torch.tensor(subj_losses)).item(),
                'obj_loss': torch.mean(torch.tensor(obj_losses)).item(),
                'accuracy': accuracy,
                'subj_acc': correct_subj / examples,
                'obj_acc': correct_obj / examples
            })

        # After each epoch, append information about the run to our lists
        subj = torch.mean(torch.tensor(subj_losses))
        obj = torch.mean(torch.tensor(obj_losses))
        epoch_losses.append((subj.item(), obj.item()))
        accuracies.append(round(sum(running_accs) / len(running_accs) * 100, 2))
        baseline_acc.append(round(sum(running_baseline) / len(running_baseline) * 100, 2))

    progress.close()

    # In the end, print out our information on a per-epoch basis
    print("Training finished.")
    print("")
    print("TRAINING REPORT")
    print("===============")
    print("")
    print("    Epoch | Subject losses | Object losses | Accuracy | Baseline Accuracy")
    print("    ------+----------------+---------------+----------+------------------")
    for idx in range(n_epochs):
        epoch = str(idx + 1).rjust(5)
        subj_loss = str(round(epoch_losses[idx][0], 2)).rjust(14)
        obj_loss = str(round(epoch_losses[idx][1], 2)).rjust(13)
        acc = (str(accuracies[idx]) + " %").rjust(8)
        base = (str(baseline_acc[idx]) + " %").rjust(17)
        print(f"    {epoch} | {subj_loss} | {obj_loss} | {acc} | {base}")

    # Print out a last mean line
    all_subject_losses, all_object_losses = zip(*epoch_losses)
    subj_mean = (str(round(sum(all_subject_losses) / len(epoch_losses), 2))).rjust(14)
    obj_mean = (str(round(sum(all_object_losses) / len(epoch_losses), 2))).rjust(13)
    acc_mean = (str(round(sum(accuracies) / len(accuracies), 2)) + " %").rjust(8)
    base_mean = (str(round(sum(baseline_acc) / len(baseline_acc), 2)) + " %").rjust(17)
    print("    ------+----------------+---------------+----------+------------------")
    print(f"     Mean | {subj_mean} | {obj_mean} | {acc_mean} | {base_mean}")

    # BEGIN: WRITE REPORT FILE =================================================
    formatted_date = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    print(f"Writing report to file output/report_{formatted_date}.tsv")
    with open(f"output/report_{formatted_date}.tsv", "w+") as fp:
        fp.write(f"Epoch\tSubject losses\tObject losses\tAccuracy\tBaseline Accuracy\tBatch size\tLearning rate\tWord vocabulary size\tContexts\n")
        for idx in range(n_epochs):
            fp.write(f"{idx + 1}\t{epoch_losses[idx][0]}\t{epoch_losses[idx][1]}\t{accuracies[idx]}\t{baseline_acc[idx]}\t{batch_size}\t{lr}\t{len(word_vocab)}\t{len(contexts_vocab)}\n")
    # END: WRITE REPORT FILE ===================================================
    print("Report written. Instantiating predictor and returning from training loop.")
    predictor = Predictor(subject_model, object_model, labels)
    print("Training done.")

    return predictor # Return the trained models
