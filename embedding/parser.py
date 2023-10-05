import numpy as np


class Rule:
    def __init__(self, length, pattern, replacement):
        self.length = length
        self.pattern = pattern
        self.replace = replacement


class Parser:
    def __init__(self, schema, rules, next_a, args):
        self.schema = schema
        self.rules = rules
        self.next_a = next_a
        self.args = args
        self.dim = rules[0].pattern.shape[0]

    def parse(self, buffer, depth=10):
        for i in range(depth):
            new_buffer = self.parse_step(buffer)
            # If no changes, return
            if len(new_buffer) == len(buffer):
                return new_buffer
            buffer = new_buffer
        return buffer

    def parse_step(self, buffer):
        for rule in self.rules:
            rl = rule.length
            p = rule.pattern

            new_buffer = []
            i = 0
            while i < len(buffer) - rl + 1:
                res = np.zeros(self.dim)
                M = np.identity(self.dim)
                # Encode buf
                buf = buffer[i:(i + rl)]
                for s in buf:
                    res += M @ s
                    M = M @ self.next_a
                # Check if match
                if p.T @ res > np.linalg.norm(p)**2 - .5:
                    attrs = np.zeros(self.dim)
                    for x, a in zip(buf, self.args):
                        attrs += a @ x
                    new_buffer.append(rule.replace + attrs)
                    i += rl
                else:
                    new_buffer.append(buf[0])
                    i += 1
            new_buffer += buffer[i:]
            buffer = new_buffer
        return buffer
