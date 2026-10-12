"""Check weight cleaning and fitted preprocessing without running the full CLI."""

import ast
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


def pipeline_functions():
    # Load declarations only: importing the CLI directly would start training.
    path = Path(__file__).resolve().parents[1] / 'freight_pipeline.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    constants = {'CORE', 'FULL', 'SEED', 'MAX_ITER'}
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constants for target in node.targets
        ):
            nodes.append(node)
    scope = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def loads(weights):
    count = len(weights)
    return pd.DataFrame({
        'load_id': [f'ROW-{i}' for i in range(count)],
        'posted_rate': [500.] * count,
        'pickup': ['Lexington'] * count,
        'delivery': ['Fort Wayne'] * count,
        'distance': [360.] * count,
        'equipment': ['Dry Van'] * count,
        'weight': weights,
        'date': ['2025-07-01'] * count,
        'pickup_lat': [38.] * count, 'pickup_lon': [-84.] * count,
        'delivery_lat': [41.] * count, 'delivery_lon': [-85.] * count,
        'market_index': [1.] * count, 'quote_signal': [2.] * count,
    })


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = pipeline_functions()

    def test_invalid_weights_and_derived_values_are_missing(self):
        frame = loads([1000., -2000., 0., np.nan])
        original = frame.copy(deep=True)
        for mode in ['full', 'core']:
            with self.subTest(mode=mode):
                x = self.scope['features'](frame, mode)
                self.assertEqual(x.loc[0, 'weight'], 1000.)
                self.assertTrue(x.loc[1:, ['weight', 'weight_per_mile']].isna().all().all())
                self.assertNotIn('load_id', x)
                self.assertNotIn('posted_rate', x)
        pd.testing.assert_frame_equal(frame, original)

    def test_imputation_uses_training_values_and_handles_new_categories(self):
        train = loads([1000., 3000., -2000., 0., np.nan])
        future = loads([-9000., 100000.])
        future['pickup'] = 'Unseen pickup'
        future['market_index'] = np.nan
        for mode in ['full', 'core']:
            with self.subTest(mode=mode):
                x = self.scope['features'](train, mode)
                model = self.scope['build_model'](x, 'median').fit(x, train.posted_rate)
                preprocessor = model.named_steps['preprocess']
                columns = preprocessor.transformers_[0][2]
                imputer = preprocessor.named_transformers_['numeric'].named_steps['impute']
                self.assertEqual(imputer.statistics_[columns.index('weight')], 2000.)
                before = imputer.statistics_.copy()
                transformed = preprocessor.transform(self.scope['features'](future, mode))
                self.assertTrue(np.isfinite(transformed).all())
                np.testing.assert_array_equal(imputer.statistics_, before)


if __name__ == '__main__':
    unittest.main()
