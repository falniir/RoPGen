import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from typing import Tuple, List
from sklearn.preprocessing import LabelEncoder

from data_loading.UtilityEntities import Path, NodeType, PathContexts, PathContext
from util import ProcessedSnapshotFolder


class PathMinerSnapshotLoader:

    def __init__(self, project_folder: ProcessedSnapshotFolder):
        self._tokens = self._load_tokens(project_folder.tokens_file)
        self._node_types = self._load_node_types(project_folder.node_types_file)
        self._paths = self._load_paths(project_folder.paths_file)
        _, self._path_contexts = self._load_path_contexts_files(project_folder.path_contexts_file)
        self._original_labels, self._tokens_by_author, self._paths_by_author = self._load_rf_contexts_file(project_folder.path_tokens_file)
        self._original_labels, self._labels,self._train_labels,self._test_labels,self._test_train_dict = self.enumerate_labels(self._original_labels)


        entities, counts = np.unique(self._labels, return_counts=True)
        # for i, orig in enumerate(self._original_labels):
        #     print(f'{orig} -> {i}')
        # ec = [(c, e) for e, c in zip(entities, counts)]
        # for i, (c, e) in enumerate(sorted(ec)):
        #     print(f'{i}: {e} -> {c} | {c / len(self._labels):.4f}')
        self._n_classes = len(entities)

    def _load_tokens(self, tokens_file: str) -> np.ndarray:
        # return self._series_to_ndarray(
        #     pd.read_csv(tokens_file, sep=',', index_col='id', usecols=['id', 'token'], squeeze=True)
        # )
        tokens = self._load_stub(tokens_file, 'token')
        return self._series_to_ndarray(tokens)

    def _load_paths(self, paths_file: str) -> np.ndarray:
        # paths = pd.read_csv(paths_file, sep=',', index_col='id', usecols=['id', 'path'], squeeze=True)
        paths = self._load_stub(paths_file, 'path')
        paths = paths.map(
            lambda nt: Path(
                list(map(int, nt.split()))
            )
        )
        return self._series_to_ndarray(paths)

    def _load_node_types(self, node_types_file: str) -> np.ndarray:
        # node_types = pd.read_csv(node_types_file, sep=',', index_col='id', usecols=['id', 'node_type'], squeeze=True)
        node_types = self._load_stub(node_types_file, 'node_type')
        node_types = node_types.map(lambda nt: NodeType(*nt.split()))
        return self._series_to_ndarray(node_types)

    @staticmethod
    def _load_stub(filename: str, col_name: str) -> pd.Series:
        # Reading CSV with proper quoting and lineterminator handling
        df = pd.read_csv(filename, sep=',', quoting=3, engine='python')



        
        # Fill missing values in the 'token' column (if any)
        df[col_name].fillna('', inplace=True)
        
        # Debugging: Print available columns and first few rows
        print(f"Available columns in {filename}: {df.columns}")
        print(df.head())  # Print first few rows for validation
        
        # Set 'id' column as index
        df = df.set_index('id')
        
        # Return the 'token' column as a pandas Series
        if col_name not in df.columns:
            raise KeyError(f"Column '{col_name}' not found in {filename}. Available columns: {df.columns}")
        return df[col_name]


    @staticmethod
    def _load_rf_contexts_file(rf_contexts_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_data = [line.strip().split(' ', 1) for line in open(rf_contexts_file, 'r').readlines()]
        labels = []
        tokens = []
        paths = []

        for label, contexts in tqdm(raw_data):
            labels.append(label)
            local_tokens = []
            local_paths = []
            for context in contexts.split():
                t, val = context.split(',')
                if t == 'token':
                    local_tokens.append(int(val))
                else:
                    local_paths.append(int(val))
            tokens.append(np.array(local_tokens))
            paths.append(np.array(local_paths))

        return np.array(labels), np.array(tokens), np.array(paths)

    @staticmethod
    def _load_path_contexts_files(path_contexts_file: str) -> Tuple[np.ndarray, PathContexts]:

        raw_data = [line.strip().split(' ', 1) for line in open(path_contexts_file, 'r').readlines()]

        labels = np.array([d[0] for d in raw_data])
        raw_contexts = [d[1] if len(d) == 2 else "1,1,1" for d in raw_data]

        path_contexts = [
            np.array(list(map(
                lambda ctx: PathContext.fromstring(ctx, sep=','),
                raw_context.split(' ')
            )), dtype=np.object)
            for raw_context in raw_contexts
        ]

        starts = np.array(list(map(
            lambda ctx_array: np.fromiter(map(lambda ctx: ctx.start_token, ctx_array), np.int32,
                                          count=ctx_array.size),
            path_contexts
        )), dtype=np.object)

        paths = np.array(list(map(
            lambda ctx_array: np.fromiter(map(lambda ctx: ctx.path, ctx_array), np.int32, count=ctx_array.size),
            path_contexts
        )), dtype=np.object)

        ends = np.array(list(map(
            lambda ctx_array: np.fromiter(map(lambda ctx: ctx.end_token, ctx_array), np.int32, count=ctx_array.size),
            path_contexts
        )), dtype=np.object)

        return labels, PathContexts(starts, paths, ends)

    @staticmethod
    def enumerate_labels(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        tmp_labels = label_encoder.transform(labels)
        print('============================')
        labels = labels.tolist()
        #print(labels)
        #print(tmp_labels)
        train_labels = []
        test_labels = []
        test_train_dict = {}
        for i in range(len(labels)):
            if labels[i].find("train_")==0:
                train_labels.append(tmp_labels[i])
            else:
                test_labels.append(tmp_labels[i])
            if labels[i].find("test_")==0:
                test_name = labels[i].split('_')[-1]
                if test_train_dict.get(tmp_labels[i],-1) == -1:
                    train_index = labels.index("train_"+test_name)
                    test_train_dict[tmp_labels[i]] = tmp_labels[train_index]

        return label_encoder.classes_, tmp_labels,list(set(train_labels)),list(set(test_labels)),test_train_dict

    @staticmethod
    def _series_to_ndarray(series: pd.Series) -> np.ndarray:
        converted_values = np.empty(max(series.index) + 1, dtype=np.object)
        for ind, val in zip(series.index, series.values):
            converted_values[ind] = val
        return converted_values

    def tokens(self) -> np.ndarray:
        return self._tokens

    def paths(self) -> np.ndarray:
        return self._paths

    def node_types(self) -> np.ndarray:
        return self._node_types

    def train_labels(self):
        return self._train_labels
    def test_labels(self):
        return self._test_labels   
    def test_train_dict(self):
        return self._test_train_dict 

    def original_labels(self) -> np.ndarray:
        return self._original_labels


    def labels(self) -> np.ndarray:
        return self._labels

    def path_contexts(self) -> PathContexts:
        return self._path_contexts

    def n_classes(self) -> int:
        return self._n_classes
