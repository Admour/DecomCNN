import numpy as np
from E2USD.utils import reorder_label

class E2USD:
    def __init__(self, win_size, step, encoder, clustering_component, verbose=False):
        self.__win_size = win_size
        self.__step = step
        self.__offset = int(win_size/2)
        self.__encoder = encoder
        self.__clustering_component = clustering_component

    def fit(self, X, win_size, step):

        self.__length = X.shape[0]
        self.fit_encoder(X)
        self.__encode(X, win_size, step)
        self.__cluster()
        self.__assign_label()
        return self

    def predict(self, X, win_size, step):
        self.__length = X.shape[0]
        self.__step = step
        self.__encode(X, win_size, step)
        self.__cluster()
        self.__assign_label()
        return self

    def set_step(self, step):
        self.__step = step

    def set_clustering_component(self, clustering_obj):
        self.__clustering_component = clustering_obj
        return self

    def fit_encoder(self, X):
        self.__encoder.fit(X)
        return self

    def predict_without_encode(self, X, win_size, step):
        self.__cluster()
        self.__assign_label()
        return self

    def __encode(self, X, win_size, step):
        self.__embeddings = self.__encoder.encode(X, win_size, step)

    def __cluster(self):
        self.__embedding_label = reorder_label(self.__clustering_component.fit(self.__embeddings))

    def __assign_label(self):
        hight = len(set(self.__embedding_label))
        weight_vector = np.ones(shape=(2*self.__offset)).flatten()
        self.__state_seq = self.__embedding_label
        vote_matrix = np.zeros((self.__length,hight))
        i = 0
        for l in self.__embedding_label:
            vote_matrix[i:i+self.__win_size,l]+= weight_vector
            i+=self.__step
        self.__state_seq = np.array([np.argmax(row) for row in vote_matrix])

    def save_encoder(self):
        pass
        
    def online_threshold_cluster(self, X, win_size, step, tau, ratio):
        self.__length = X.shape[0]
        self.__step_threshold = step

        miner=1-self.delta
        maxer=1+self.delta*ratio
        label=[]
        total_clusetring  = 0
        for i in range(0,self.__length-win_size, step):
            now_x=X[i:i+win_size]
            now_win_embedding = self.__encode_one(now_x)
            if self.last_win_embedding is None:
                self.last_win_embedding = now_win_embedding
                self.last_win_state = self.__cluster_one(now_win_embedding)
                label.append(self.last_win_state)
                total_clusetring+=1
            else:
                similarity = np.dot(self.last_win_embedding,now_win_embedding.T)

                if similarity >=tau:
                    label.append(self.last_win_state)
                    tau=tau*maxer
                else:
                    new_win_state = self.__cluster_one(now_win_embedding)
                    if new_win_state != self.last_win_state:
                        self.last_win_embedding = now_win_embedding
                        self.last_win_state = new_win_state
                        tau = tau * maxer
                    else:
                        tau = tau * miner
                    label.append(self.last_win_state)
                    total_clusetring += 1
        label_np= np.array(label)
        self.threshold_label = label_np
        self.__assign_label_threshold()
        return label_np, total_clusetring

    def load_encoder(self):
        pass

    def save_result(self, path):
        pass

    def load_result(self, path):
        pass

    def plot(self, path):
        pass

    @property
    def embeddings(self):
        return self.__embeddings

    @property
    def state_seq(self):
        return self.__state_seq
    
    @property
    def embedding_label(self):
        return self.__embedding_label

    @property
    def velocity(self):
        return self.__velocity

    @property
    def change_points(self):
        return self.__change_points
