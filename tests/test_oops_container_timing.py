import io
import struct
import unittest

from scripts.audit_oops_container_timing import boxes, duration_header


class ContainerTimingTests(unittest.TestCase):
    def test_duration_header_versions(self):
        a = b'\0' * 12 + struct.pack('>II', 1000, 6316)
        b = b'\x01' + b'\0' * 19 + struct.pack('>IQ', 48000, 303168)
        self.assertEqual(duration_header(a)['seconds'], 6.316)
        self.assertEqual(duration_header(b)['seconds'], 6.316)
        with self.assertRaises(RuntimeError): duration_header(b'\0')
        with self.assertRaises(RuntimeError): duration_header(b'\0' * 20)

    def test_atom_bounds_extended_size_and_to_eof(self):
        value = struct.pack('>I4sQ', 1, b'free', 16) + struct.pack('>I4s', 0, b'free')
        self.assertEqual(list(boxes(io.BytesIO(value), 0, len(value))),
                         [(b'free', 16, 16), (b'free', 24, 24)])
        for value in (b'\0', struct.pack('>I4s', 100, b'moov'), struct.pack('>I4s', 4, b'moov')):
            with self.assertRaises(RuntimeError): list(boxes(io.BytesIO(value), 0, len(value)))


if __name__ == '__main__':
    unittest.main()
