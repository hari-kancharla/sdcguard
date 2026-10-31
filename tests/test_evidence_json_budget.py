"""Small raw node-admission controls. Owner: Hari Krishna Kancharla."""
from collections import UserDict
import json
import unittest
from unittest.mock import patch

from sdcguard import config_inputs
from sdcguard import evidence_io as io
from sdcguard import evidence_json_budget as budget


def decoded_nodes(raw):
    count = [0]
    config_inputs._json_values(json.loads(raw), count=count, max_nodes=10000)
    return count[0]


class JsonBudgetTests(unittest.TestCase):
    def test_valid_roots_containers_keys_and_primitives_match_existing_counter(self):
        for value in (None, True, False, 0, -12, 1.25e-8, '', 'text', [], {},
                      [None, True, 2, 's'], {'a': [None, True, 2]}, {'k': {'s': 'x'}}):
            with self.subTest(value=value):
                raw = json.dumps(value, ensure_ascii=False).encode('utf8')
                self.assertEqual(budget.count_members({'input.json': raw}), decoded_nodes(raw))
        self.assertEqual(budget.count_members({'input.json': b'{"a":[null,true,2]}'}), 6)

    def test_escaped_quotes_backslashes_and_structural_characters_in_strings(self):
        values = ['quote"', 'backslash\\', '\\"', '\\\\"', '[]{}:, true null',
                  {'a"\\': ['slash\\end\\', 'line\nreturn\rtab\t', '\\u1234']}]
        for value in values:
            with self.subTest(value=value):
                raw = json.dumps(value).encode('utf8')
                self.assertEqual(budget.count_members({'input.json': raw}), decoded_nodes(raw))

    def test_utf8_scalar_content_and_escaped_nul_are_counted_without_decoding(self):
        value = {'λ': ['😀', 'e\u0301', '\ufeff', '\x00', '\u07ff', '\u0800', '\uffff', '\U0010ffff']}
        raw = json.dumps(value, ensure_ascii=False).encode('utf8')
        self.assertEqual(budget.count_members({'unicode.json': raw}), decoded_nodes(raw))
        self.assertEqual(budget.count_members({'quoted-bom.json': b'"\xef\xbb\xbf"'}), 1)

    def test_jsonl_cr_lf_crlf_and_whitespace_use_one_total(self):
        raw = b'{"a":1}\r[true,null]\r\n"x"\n\n0\r'
        expected = sum(decoded_nodes(line) for line in raw.splitlines() if line.strip())
        self.assertEqual(budget.count_members({'events.jsonl': raw}), expected)
        self.assertEqual(expected, 8)

    def test_unfinished_jsonl_quote_or_escape_cannot_hide_later_records(self):
        self.assertEqual(budget.count_members({'events.jsonl': b'"unfinished\n[1,2]\r{"a":true}\r\nnull\n'}), 8)
        self.assertEqual(budget.count_members({'events.jsonl': b'"unfinished\\\r[1]\n[2]\r\n'}), 5)
        with self.assertRaisesRegex(io.EvidenceError, 'node allowance'):
            budget.count_members({'events.jsonl': b'"unfinished\n[1,2]\n{"a":true}\n'}, max_nodes=4)

    def test_partial_malformed_records_are_bounded_without_semantic_validation(self):
        self.assertEqual(budget.count_members({'partial.jsonl': b'{"a":'}), 2)
        self.assertEqual(budget.count_members({'malformed.jsonl': b'1[2]\ntrue:false\n'}), 5)
        self.assertEqual(budget.count_members({'duplicate.json': b'{"a":0,"a":1}'}), 5)
        self.assertEqual(budget.count_members({'punctuation.json': b'],: '}), 0)

    def test_multiple_individually_small_files_share_one_allowance(self):
        files = {'inputs/first/report.json': b'{"a":0}', 'inputs/second/events.jsonl': b'[1,2]\n'}
        self.assertEqual(budget.count_members(files, max_nodes=6), 6)
        with self.assertRaisesRegex(io.EvidenceError, 'node allowance'):
            budget.count_members(files, max_nodes=5)

    def test_default_uses_current_global_allowance(self):
        with patch.object(io, 'MAX_JSON_NODES', 2):
            with self.assertRaisesRegex(io.EvidenceError, 'node allowance'):
                budget.count_members({'input.json': b'[1,2]'})
            self.assertEqual(budget.count_members({'input.json': b'[1]'}), 2)

    def test_explicit_recording_budget_is_finite_and_default_stays_smaller(self):
        self.assertEqual(budget.MAX_TOTAL_JSON_NODES, 2_000_000)
        files = {'events.jsonl': b'0\n' * budget.MAX_TOTAL_JSON_NODES}
        self.assertEqual(budget.count_members(files, max_nodes=budget.MAX_TOTAL_JSON_NODES),
                         budget.MAX_TOTAL_JSON_NODES)
        with self.assertRaisesRegex(io.EvidenceError, 'node allowance exceeded'):
            budget.count_members(files)
        files['next.json'] = b'0'
        with self.assertRaisesRegex(io.EvidenceError, 'node allowance exceeded'):
            budget.count_members(files, max_nodes=budget.MAX_TOTAL_JSON_NODES)

    def test_larger_aggregate_admission_does_not_raise_per_document_limit(self):
        raw = b'[' + b'0,' * (io.MAX_JSON_NODES - 1) + b'0]'
        self.assertEqual(budget.count_members({'input.json': raw},
                         max_nodes=budget.MAX_TOTAL_JSON_NODES), io.MAX_JSON_NODES + 1)
        with self.assertRaises(io.EvidenceError):
            io.parse_json(raw)

    def test_zero_allowance_and_unstructured_members(self):
        files = {'raw.txt': b'\xff\x00', 'script.py': b'{}[]', 'index.html': b'{}', 'upper.JSON': b'[1]',
                 'empty.json': b'', 'space.jsonl': b' \t\r\n'}
        self.assertEqual(budget.count_members(files, max_nodes=0), 0)
        self.assertEqual(budget.count_members({}, max_nodes=0), 0)
        with self.assertRaisesRegex(io.EvidenceError, 'node allowance'):
            budget.count_members({'input.json': b'0'}, max_nodes=0)

    def test_stops_at_cap_before_later_member_or_later_encoding(self):
        files = {'first.json': b'[0,1]', 'later.json': b'\xff'}
        real = budget._count_member
        calls = []
        def count(raw, jsonl, allowance):
            calls.append(raw)
            return real(raw, jsonl, allowance)
        with patch.object(budget, '_count_member', count):
            with self.assertRaisesRegex(io.EvidenceError, 'node allowance'):
                budget.count_members(files, max_nodes=2)
        self.assertEqual(calls, [files['first.json']])
        with self.assertRaisesRegex(io.EvidenceError, 'node allowance'):
            budget.count_members({'first.json': b'[0,1,\xff]'}, max_nodes=2)

    def test_no_json_decoder_or_value_tree_walk_is_invoked(self):
        def forbidden(*args, **kwargs):
            raise AssertionError('raw admission must not decode JSON')
        with patch.object(json, 'loads', forbidden), patch.object(json.JSONDecoder, 'decode', forbidden), \
                patch.object(json.JSONDecoder, 'raw_decode', forbidden), patch.object(config_inputs, '_json_values', forbidden):
            self.assertEqual(budget.count_members({'input.json': b'{"x":[true,null]}', 'events.jsonl': b'0\n'}), 6)

    def test_unsupported_boms_utf16_utf32_and_nul_refuse(self):
        cases = [b'\xef\xbb\xbf{}', b'\xff\xfe{}', b'\xfe\xff{}', b'\x00\x00\xfe\xff{}',
                 b'\xff\xfe\x00\x00{}', b'{}\x00', b'"\x00"']
        cases.extend('{"x":1}'.encode(encoding) for encoding in ('utf-16', 'utf-16-le', 'utf-16-be', 'utf-32', 'utf-32-le', 'utf-32-be'))
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaisesRegex(io.EvidenceError, 'unsupported structured encoding'):
                budget.count_members({'input.json': raw})
        for prefix in (b'\n', b'0\r', b'0\r\n'):
            with self.subTest(prefix=prefix), self.assertRaisesRegex(io.EvidenceError, 'unsupported structured encoding'):
                budget.count_members({'events.jsonl': prefix + b'\xef\xbb\xbf{}\n'})

    def test_malformed_overlong_surrogate_out_of_range_and_truncated_utf8_refuse(self):
        for raw in (b'"\x80"', b'"\xc0\x80"', b'"\xc2"', b'"\xe0\x80\x80"',
                    b'"\xed\xa0\x80"', b'"\xf0\x80\x80\x80"', b'"\xf4\x90\x80\x80"',
                    b'"\xf5\x80\x80\x80"', b'"\xe2\x82', b'"\xf0\x90\x80'):
            with self.subTest(raw=raw), self.assertRaisesRegex(io.EvidenceError, 'unsupported structured encoding'):
                budget.count_members({'input.json': raw})
        with self.assertRaisesRegex(io.EvidenceError, 'unsupported structured encoding'):
            budget.count_members({'events.jsonl': b'"\xe2\n0\n'})

    def test_strict_argument_types_relative_names_and_allowance_ranges(self):
        class DictSubclass(dict):
            pass
        for files in ([], (), UserDict(), DictSubclass()):
            with self.subTest(files=type(files)), self.assertRaises(io.EvidenceError):
                budget.count_members(files)
        for limit in (False, True, -1, 1.0, '1', budget.MAX_TOTAL_JSON_NODES + 1):
            with self.subTest(limit=limit), self.assertRaises(io.EvidenceError):
                budget.count_members({}, max_nodes=limit)
        for name in ('', '../outside.json', '/outside.json', 'nested/../x.json', 'a\\x.json', 'drive:x.json', 7):
            with self.subTest(name=name), self.assertRaises(io.EvidenceError):
                budget.count_members({name: b'0'})
        for raw in (bytearray(b'0'), memoryview(b'0'), '0', None):
            with self.subTest(raw=type(raw)), self.assertRaises(io.EvidenceError):
                budget.count_members({'raw.txt': raw})


if __name__ == '__main__':
    unittest.main()
