import torch
import transformers


def compute_l2_norm_matching(model, patterns):
    """
    Compute an L2 norm of all parameters whose name matches a patter from patterns args
    :param patterns: Patterns for matching parameters to be regularized
    :param model: model for which an L2 norm shall be calculated
    :return: L2 norm of the model
    """
    # TODO START
    # Compute sum of l2 norms of all parameters whose name matches at least one patter from patterns
    l2 = 0
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            l2 += torch.norm(param, p=2)

    return l2
    # TODO END


class TokenClassifierOutput:
    """
    Helper class to create common interface for HuggingFace and custom models
    because transformers.modeling_outputs.TokenClassifierOutput (output of SLAVIC and CZERT)
    is an internal class that cannot be accessed publicly.
    """

    def __init__(self, logits, loss):
        self.logits = logits
        self.loss = loss


class Czert(torch.nn.Module):
    """
    Implementation of Czert model wrapper.
    """

    def __init__(self, czert_model_path,
                 device,
                 random_init=False,
                 dropout_probs=0.2,
                 freeze_embedding_layer=False,
                 freeze_first_x_layers=0,
                 num_labels=1,):
        """
        Constructor
        :param czert_model_path: path to a folder with the model or a huggingface identifier of the model
        :param random_init: flag indicating whether to randomly initialize the model
        :param dropout_probs: dropout probability used by the model
        :param freeze_embedding_layer: flag indicating whether an embedding layer shall be frozen during the training
        :param freeze_first_x_layers: number of bottom layers of the model to be frozen during the training
        :param num_labels: number of labels of the task
        """
        super().__init__()

        self.__device = device

        # Load BART model pretrained for generating source code
        self.__czert_config = transformers.BertConfig.from_pretrained(czert_model_path)
        self.__czert_config.hidden_dropout_prob = dropout_probs
        self.__czert_config.num_labels = num_labels

        # Load pretrained Czert model
        if not random_init:
            self._czert_model: transformers.BertForTokenClassification = \
                transformers.BertForTokenClassification.from_pretrained(czert_model_path, config=self.__czert_config)
        else:
            self._czert_model = transformers.BertForTokenClassification(config=self.__czert_config)

        if freeze_first_x_layers > 0:
            self.__freeze_x_bottom_layers(freeze_first_x_layers)

        if freeze_embedding_layer:
            self.__freeze_embedding_layer()

    def get_config(self):
        """
        Get HuggingFace model config
        :return: config of the Huggingface model
        """
        return self.__czert_config

    def __freeze_x_bottom_layers(self, num_freeze_layers):
        """
        Freeze all X bottom transformer layers of the encoder stack
        :param num_freeze_layers: number of bottom layers to be frozen
        :return: N/A
        """
        freeze_layers = [f"layer.{i}." for i in range(num_freeze_layers)]
        for name, param in self._czert_model.named_parameters():
            if any(layer_pattern in name for layer_pattern in freeze_layers):
                param.requires_grad = False

    def __freeze_embedding_layer(self):
        """
        Freeze embedding layer weights
        :return: N/A
        """
        for name, param in self._czert_model.named_parameters():
            if "embedding" in name:
                param.requires_grad = False

    def compute_l2_norm(self):
        """
        Compute L2 norm of the model
        :return: L2 norm value
        """
        return torch.tensor([0.0]).to(self.__device)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                labels,
                ):
        """
        Calculate a forward pass of the Czert model
        :param input_ids: IDs of the tokenized input sequence
        :param attention_mask: attention mask for the encoder (to mask out [SEP] and [PAD])
        :param token_type_ids: token type IDs for the encoder (all zeros if the model only a single sentence)
        :param labels: expected labels
        :return: predictions and loss
        """
        # Build inputs for the czert model
        inputs = {"input_ids": input_ids,
                  "attention_mask": attention_mask,
                  "token_type_ids": token_type_ids,
                  "labels": labels,
                  }

        # Calculate activations of the Czert model
        return self._czert_model(**inputs)


class Slavic(torch.nn.Module):
    """
    Implementation of Slavic model wrapper.
    """

    def __init__(self, slavic_model_path,
                 device,
                 random_init=False,
                 dropout_probs=0.2,
                 freeze_embedding_layer=False,
                 freeze_first_x_layers=0,
                 num_labels=1):
        """
        Constructor
        :param slavic_model_path: path to a folder with the model or a huggingface identifier of the model
        :param random_init: flag indicating whether to randomly initialize the model
        :param dropout_probs: dropout probability used by the model
        :param freeze_embedding_layer: flag indicating whether an embedding layer shall be frozen during the training
        :param freeze_first_x_layers: number of bottom layers of the model to be frozen during the training
        :param num_labels: number of labels of the task
        """
        super().__init__()

        self.__device = device

        # Load BART model pretrained for generating source code
        self.__slavic_model_config = transformers.BertConfig.from_pretrained(slavic_model_path)
        self.__slavic_model_config.hidden_dropout_prob = dropout_probs
        self.__slavic_model_config.num_labels = num_labels

        # Load pretrained Slavic model
        if not random_init:
            self._slavic_model: transformers.AutoModelForTokenClassification = \
                transformers.AutoModelForTokenClassification.from_pretrained(slavic_model_path,
                                                                             config=self.__slavic_model_config)
        else:
            self._slavic_model = transformers.AutoModelForTokenClassification(config=self.__slavic_model_config)

        if freeze_first_x_layers > 0:
            self.__freeze_x_bottom_layers(freeze_first_x_layers)

        if freeze_embedding_layer:
            self.__freeze_embedding_layer()

    def get_config(self):
        """
        Get HuggingFace model config
        :return: config of the Huggingface model
        """
        return self.__slavic_model_config

    def __freeze_x_bottom_layers(self, num_freeze_layers):
        """
        Freeze all X bottom transformer layers of the encoder stack
        :param num_freeze_layers: number of bottom layers to be frozen
        :return: N/A
        """
        freeze_layers = [f"layer.{i}." for i in range(num_freeze_layers)]
        for name, param in self._slavic_model.named_parameters():
            if any(layer_pattern in name for layer_pattern in freeze_layers):
                param.requires_grad = False

    def __freeze_embedding_layer(self):
        """
        Freeze embedding layer weights
        :return: N/A
        """
        for name, param in self._slavic_model.named_parameters():
            if "embedding" in name:
                param.requires_grad = False

    def compute_l2_norm(self):
        """
        Compute L2 norm of the model
        :return: L2 norm value
        """
        return torch.tensor([0.0]).to(self.__device)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                labels,
                ):
        """
        Calculate a forward pass of the SLAVIC model
        :param input_ids: IDs of the tokenized input sequence
        :param attention_mask: attention mask for the encoder (to mask out [SEP] and [PAD])
        :param token_type_ids: token type IDs for the encoder (all zeros if the model only a single sentence)
        :param labels: expected labels
        :return: predictions and loss
        """
        # Build inputs for the czert model
        inputs = {"input_ids": input_ids,
                  "attention_mask": attention_mask,
                  "token_type_ids": token_type_ids,
                  "labels": labels,
                  }

        # Calculate activations of the Czert model
        return self._slavic_model(**inputs)


class RNN(torch.nn.Module):
    def __init__(self,
                 vocab_size,
                 device,
                 embedding_dimension=128,
                 dropout_probs=0.2,
                 freeze_embedding_layer=False,
                 num_labels=1,
                 rnn_hidden_size=128,
                 use_bias=True,
                 l2_alpha=0.05,):
        """
        Constructor
        :param dropout_probs: dropout probability used by the model
        :param freeze_embedding_layer: flag indicating whether an embedding layer shall be frozen during the training
        :param num_labels: number of labels of the task
        """
        super().__init__()

        self._num_labels = num_labels
        self.__vocab_size = vocab_size
        self.__embedding_dimension = embedding_dimension
        self.__dropout_prob = dropout_probs
        self.__use_bias = use_bias
        self.__rnn_hidden_size = rnn_hidden_size
        self.__l2_alpha = l2_alpha
        self.__device = device
        self.__hidden_state = None
        self._hidden_states = []

        # TODO START
        self._embedding_layer = torch.nn.Embedding(
            num_embeddings=self.__vocab_size,
            embedding_dim=self.__embedding_dimension
        )
        self._loss = torch.nn.CrossEntropyLoss()
        self._dropout_layer = torch.nn.Dropout(p=self.__dropout_prob)
        self._new_hidden_state_layer = torch.nn.Linear(
            in_features=self.__embedding_dimension + self.__rnn_hidden_size,
            out_features=self.__rnn_hidden_size,
            bias=self.__use_bias
        )
        self._output_layer = torch.nn.Linear(
            in_features=self.__rnn_hidden_size,
            out_features=self._num_labels,
            bias=self.__use_bias
        )
        # TODO END

        if freeze_embedding_layer:
            self.__freeze_embedding_layer()

    def get_config(self):
        """
        Get model config
        :return: config of the Huggingface model
        """
        return {"model_type": "RNN"}

    def compute_l2_norm(self):
        """
        Compute L2 norm of the model
        :return: L2 norm value
        """
        # TODO START
        return compute_l2_norm_matching(self, ["new_hidden_state_layer", "output_layer"]) * self.__l2_alpha
        # TODO END

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                labels,
                ):
        """
        Calculate a forward pass of the LSTM model
        :param input_ids: IDs of the tokenized input sequence
        :param attention_mask: attention mask for the encoder (to mask out [SEP] and [PAD]) - IGNORED
        :param token_type_ids: token type IDs for the encoder (all zeros if the model only a single sentence) - IGNORED
        :param labels: expected labels
        :return: predictions and loss
        """
        self.__init_zero_hidden(input_ids.shape[0])
        # TODO START
        # embeddings and dropouts
        # iteration over the individual tokens generating a sequence of hidden states (in a loop)
        # preserve all the hidden states
        # create a single tensor of all the hidden states
        # apply output layer with softmax over the hidden states
        embeddings = self._embedding_layer(input_ids)
        embeddings = self._dropout_layer(embeddings)
        for i in range(embeddings.shape[1]):
            current_input = embeddings[:, i, :]
            current_hidden_state = torch.tanh(self._new_hidden_state_layer(
                torch.cat([current_input, self.__hidden_state], dim=1))
            )
            current_hidden_state = self._dropout_layer(current_hidden_state)
            self.__hidden_state = current_hidden_state
            self._hidden_states.append(current_hidden_state.clone())

        self._hidden_states = torch.stack(self._hidden_states, dim=1)
        logits = self._output_layer(self._hidden_states)
        loss = self._loss(logits.view(-1, self._num_labels), labels.view(-1))
        out = TokenClassifierOutput(logits=logits, loss=loss)
        return out

    def __init_zero_hidden(self, batch_size):
        """
        Returns a hidden state with specified batch size
        """
        self.__hidden_state = torch.zeros(batch_size, self.__rnn_hidden_size, requires_grad=False)
        self.__hidden_state = self.__hidden_state.to(self.__device)
        self._hidden_states = []

    def __freeze_embedding_layer(self):
        """
        Freeze embedding layer weights
        :return: N/A
        """
        # TODO START
        # freeze embedding layer
        #  - iterate over all parameters and set x.requires_grad = False for the embedding weights
        for name, param in self.named_parameters():
            if "embedding" in name:
                param.requires_grad = False
        # TODO END


class LSTM(torch.nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dimension=128,
                 dropout_probs=0.2,
                 freeze_embedding_layer=False,
                 freeze_first_x_layers=0,
                 num_labels=1,
                 lstm_layers=2,
                 lstm_hidden_size=128,
                 use_bias=True,
                 l2_alpha=0.05):
        """
        Constructor
        :param dropout_probs: dropout probability used by the model
        :param freeze_embedding_layer: flag indicating whether an embedding layer shall be frozen during the training
        :param freeze_first_x_layers: number of bottom layers of the model to be frozen during the training
        :param num_labels: number of labels of the task
        """
        super().__init__()

        self._num_labels = num_labels
        self.__vocab_size = vocab_size
        self.__lstm_layers = lstm_layers
        self.__embedding_dimension = embedding_dimension
        self.__dropout_prob = dropout_probs
        self.__use_bias = use_bias
        self.__lstm_hidden_size = lstm_hidden_size
        self.__l2_alpha = l2_alpha

        # TODO START
        self._loss = torch.nn.CrossEntropyLoss()
        self._embedding_layer = torch.nn.Embedding(
            num_embeddings=self.__vocab_size,
            embedding_dim=self.__embedding_dimension
        )
        self._lstm = torch.nn.LSTM(  # don't forget to use BiLSTM
            input_size=self.__embedding_dimension,
            hidden_size=self.__lstm_hidden_size,
            num_layers=self.__lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self._dropout_layer = torch.nn.Dropout(p=self.__dropout_prob)
        self._dense = torch.nn.Linear(
            in_features=self.__lstm_hidden_size * 2,
            out_features=self.__lstm_hidden_size * 2,
            bias=self.__use_bias
        )
        self._classification_head = torch.nn.Linear(
            in_features=self.__lstm_hidden_size * 2,
            out_features=self._num_labels,
            bias=self.__use_bias
        )
        # TODO END

        if freeze_first_x_layers > 0:
            self.__freeze_x_bottom_layers(freeze_first_x_layers)

        if freeze_embedding_layer:
            self.__freeze_embedding_layer()

    def get_config(self):
        """
        Get model config
        :return: config of the Huggingface model
        """
        return {"model_type": "LSTM"}

    def __freeze_x_bottom_layers(self, num_freeze_layers):
        """
        Freeze all X bottom transformer layers of the encoder stack
        :param num_freeze_layers: number of bottom layers to be frozen
        :return: N/A
        """
        freeze_layers = [f"_l{i}" for i in range(num_freeze_layers)]
        for name, param in self.named_parameters():
            if any(layer_pattern in name for layer_pattern in freeze_layers):
                param.requires_grad = False

    def __freeze_embedding_layer(self):
        """
        Freeze embedding layer weights
        :return: N/A
        """
        # TODO START
        # freeze embedding layer
        #  - iterate over all parameters and set x.requires_grad = False for the embedding weights
        for name, param in self.named_parameters():
            if "embedding" in name:
                param.requires_grad = False
        # TODO END

    def compute_l2_norm(self):
        """
        Compute L2 norm of the model
        :return: L2 norm value
        """
        # TODO START
        return compute_l2_norm_matching(self, ["dense", "classification_head"]) * self.__l2_alpha
        # TODO END

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                labels,
                ):
        """
        Calculate a forward pass of the LSTM model
        :param input_ids: IDs of the tokenized input sequence
        :param attention_mask: attention mask for the encoder (to mask out [SEP] and [PAD]) - IGNORED
        :param token_type_ids: token type IDs for the encoder (all zeros if the model only a single sentence) - IGNORED
        :param labels: expected labels
        :return: predictions and loss
        """

        # TODO START
        # apply embeddings and dropouts
        # BiLSTM
        # Dropout
        # Dense layer with ReLu
        # Output layer with Softmax
        embeddings = self._embedding_layer(input_ids)
        embeddings = self._dropout_layer(embeddings)
        lstm_output, _ = self._lstm(embeddings)
        lstm_output = self._dropout_layer(lstm_output)
        lstm_output = torch.relu(self._dense(lstm_output))
        logits = self._classification_head(lstm_output)  # model outputs to be used to compute the loss
        # TODO END
        loss = self._loss(logits.view(-1, self._num_labels), labels.view(-1))
        out = TokenClassifierOutput(logits=logits, loss=loss)
        return out

