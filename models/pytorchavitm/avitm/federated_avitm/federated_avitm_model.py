from models.pytorchavitm.avitm.avitm_model import AVITM_model

class FederatedAVITM(AVITM_model):
    def __init__(self, input_size, n_components=10, model_type='prodLDA',
                 hidden_sizes=(100, 100), activation='softplus', dropout=0.2,
                 learn_priors=True, batch_size=64, lr=2e-3, momentum=0.99,
                 solver='adam', num_epochs=100, reduce_on_plateau=False, 
                 topic_prior_mean=0.0, topic_prior_variance=None,
                 num_samples=10, num_data_loader_workers=0, verbose=False):

        super().__init__(input_size, n_components, model_type, hidden_sizes,
                 activation, dropout, learn_priors, batch_size, lr, momentum,
                 solver, num_epochs, reduce_on_plateau, topic_prior_mean, 
                 topic_prior_variance, num_samples, num_data_loader_workers,
                 verbose)
        
        # Current epoch for tracking federated model
        self.current_mb = -1

        # Post-training parameters 
        self.topics = None
        self.doc_topic_distrib = None
        self.word_topic_distrib = None

        # Parameters for evaluation 
        self.frob_gt_inferred_doc_dif = 0.0
        self.max_gt_inferred_top = 0.0

    def _train_minibatch(self, X, train_loss, samples_processed, topic_doc_list):

        if self.USE_CUDA:
            X = X.cuda()
        
        # Forward pass
        self.model.zero_grad() # Update gradients to zero
        prior_mean, prior_var, posterior_mean, posterior_var, posterior_log_var, \
            word_dists, topic_words, topic_document = self.model(X)
        
        topic_doc_list.extend(topic_document)

        # Backward pass: Compute gradients
        loss = self._loss(X, word_dists, prior_mean, prior_var,
                          posterior_mean, posterior_var, posterior_log_var)
        loss.backward()
        print(type(topic_document))

        return loss, train_loss, samples_processed, topic_words, topic_doc_list


    def _optimize_on_minibatch(self, X, loss, update, train_loss, samples_processed):
        # Update gradients
        # Parameter0 = prior_mean
        # Parameter1 = prior_variance
        # Parameter2 = beta
        #self.model.prior_mean.grad = update
        self.model.beta.grad = update
        
        # Perform one step of the optimizer (SGD/Adam)
        self.optimizer.step()

        # Compute train loss
        train_loss += loss.item()
        samples_processed += X.size()[0]
        
        return train_loss, samples_processed