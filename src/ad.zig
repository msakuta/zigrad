const std = @import("std");
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;
const assert = std.debug.assert;

// const debug_print = std.debug.print;
fn debug_print(fmt: [:0]const u8, _: anytype) void {
    _ = fmt;
}

pub const TapeNode = struct {
    name: [:0]const u8,
    value: TapeValue,
    data: ?f64,
    grad: ?f64,
};

pub const TapeTerm = struct {
    tape: *Tape,
    idx: TapeIndex,
    pub fn get_name(self: *const TapeTerm) ?[:0]const u8 {
        return self.*.tape.*.nodes[self.*.idx].name;
    }
    pub fn get_data(self: *const TapeTerm) ?f64 {
        return self.*.tape.*.nodes[self.*.idx].data;
    }
    pub fn get_grad(self: *const TapeTerm) ?f64 {
        return self.*.tape.*.nodes[self.*.idx].grad;
    }
    pub fn set(self: *const TapeTerm, val: f64) void {
        var node = &self.*.tape.*.nodes[self.*.idx];
        switch (node.*.value) {
            .value => |*data| data.* = val,
            else => return,
        }
    }
    pub fn eval(self: *const TapeTerm) f64 {
        return self.*.tape.eval(self.*.idx);
    }
    pub fn derive(self: *const TapeTerm, wrt: TapeTerm) ?f64 {
        self.*.tape.clear_grad();
        return self.*.tape.derive(self.*.idx, wrt.idx);
    }
    pub fn backward(self: *const TapeTerm) void {
        self.*.tape.clear_grad();
        return self.*.tape.backward(self.*.idx);
    }
    pub fn gen_graph(self: *const TapeTerm, wrt: TapeTerm, allocator: *Allocator) !?TapeTerm {
        try expect(self.*.tape == wrt.tape);
        return self.*.tape.gen_graph(self.*.idx, wrt.idx, allocator);
    }
    pub fn add(self: *const TapeTerm, rhs: TapeTerm, arena: *std.mem.Allocator) !TapeTerm {
        return self._bin_op(rhs, arena, TapeValue{ .add = .{ self.*.idx, rhs.idx } }, " - ", true);
    }
    pub fn sub(self: *const TapeTerm, rhs: TapeTerm, arena: *std.mem.Allocator) !TapeTerm {
        return self._bin_op(rhs, arena, TapeValue{ .sub = .{ self.*.idx, rhs.idx } }, " - ", true);
    }
    pub fn mul(self: *const TapeTerm, rhs: TapeTerm, arena: *std.mem.Allocator) !TapeTerm {
        return self._bin_op(rhs, arena, TapeValue{ .mul = .{ self.*.idx, rhs.idx } }, " * ", false);
    }
    pub fn div(self: *const TapeTerm, rhs: TapeTerm, arena: *std.mem.Allocator) !TapeTerm {
        return self._bin_op(rhs, arena, TapeValue{ .div = .{ self.*.idx, rhs.idx } }, " / ", false);
    }
    pub fn neg(self: *const TapeTerm, arena: *std.mem.Allocator) !TapeTerm {
        var tape = self.*.tape;
        const term = tape.*.count;
        assert(term < tape.*.nodes.len);
        tape.*.nodes[term] = TapeNode{
            .name = try std.mem.concatWithSentinel(arena.*, u8, &[_][:0]const u8{ "-", tape.*.nodes[self.*.idx].name }, 0),
            .value = TapeValue{ .neg = self.*.idx },
            .data = null,
            .grad = null,
        };
        tape.*.count += 1;
        return TapeTerm{
            .tape = tape,
            .idx = @intCast(term),
        };
    }
    pub fn apply(self: *const TapeTerm, arena: *std.mem.Allocator, name: [:0]const u8, f: *const fn (f64) f64, grad: *const fn (f64) f64, gen_graph_: ?GenGraph) !TapeTerm {
        var tape = self.*.tape;
        const term = tape.*.count;
        assert(term < tape.*.nodes.len);
        tape.*.nodes[term] = TapeNode{
            .name = try std.mem.concatWithSentinel(arena.*, u8, &[_][:0]const u8{ name, "(", tape.*.nodes[self.*.idx].name, ")" }, 0),
            .value = TapeValue{ .unary_fn = .{ .idx = self.*.idx, .f = f, .grad = grad, .gen_graph = gen_graph_ } },
            .data = null,
            .grad = null,
        };
        tape.*.count += 1;
        return TapeTerm{
            .tape = tape,
            .idx = @intCast(term),
        };
    }
    fn _bin_op(self: *const TapeTerm, rhs: TapeTerm, arena: *std.mem.Allocator, variant: TapeValue, op_name: [:0]const u8, paren: bool) !TapeTerm {
        var tape = self.*.tape;
        const term = tape.*.count;
        assert(term < tape.*.nodes.len);
        tape.*.nodes[term] = TapeNode{
            .name = if (paren) try concatWithOpParen(arena, tape.*.nodes[self.*.idx].name, op_name, tape.*.nodes[rhs.idx].name) else try concatWithOp(arena, tape.*.nodes[self.*.idx].name, op_name, tape.*.nodes[rhs.idx].name),
            .value = variant,
            .data = null,
            .grad = null,
        };
        tape.*.count += 1;
        return TapeTerm{
            .tape = tape,
            .idx = @intCast(term),
        };
    }
    pub fn from(tape: *Tape, idx: TapeIndex) TapeTerm {
        return TapeTerm{
            .tape = tape,
            .idx = idx,
        };
    }
};

pub const Tape = struct {
    nodes: []TapeNode,
    count: usize,
    pub fn new(allocator: *Allocator, max_count: usize) !Tape {
        var tape = Tape{
            .nodes = try allocator.alloc(TapeNode, max_count),
            .count = 2,
        };

        tape.nodes[0] = TapeNode{
            .name = "0",
            .value = TapeValue{ .value = 0 },
            .data = null,
            .grad = null,
        };
        tape.nodes[1] = TapeNode{
            .name = "1",
            .value = TapeValue{ .value = 1 },
            .data = null,
            .grad = null,
        };

        return tape;
    }
    pub fn variable(self: *Tape, name: [:0]const u8, v: f64) TapeTerm {
        const term = self.*.count;
        self.*.nodes[term] = TapeNode{
            .name = name,
            .value = TapeValue{ .value = v },
            .data = null,
            .grad = null,
        };
        self.*.count += 1;
        return TapeTerm{
            .tape = self,
            .idx = @as(u32, @intCast(term)),
        };
    }
    fn eval(self: *Tape, term: TapeIndex) f64 {
        const node = &self.*.nodes[term];
        if (node.*.data) |data| {
            // std.debug.print("Returning cache {s} = {?}\n", .{ node.*.name, data });
            return data;
        }
        const ret = switch (node.value) {
            .value => |v| v,
            .add => |args| self.eval(args[0]) + self.eval(args[1]),
            .sub => |args| self.eval(args[0]) - self.eval(args[1]),
            .mul => |args| self.eval(args[0]) * self.eval(args[1]),
            .div => |args| self.eval(args[0]) / self.eval(args[1]),
            .neg => |arg| -self.eval(arg),
            .unary_fn => |args| args.f(self.eval(args.idx)),
        };
        node.*.data = ret;
        return ret;
    }
    fn derive(self: *Tape, term: TapeIndex, wrt: TapeIndex) f64 {
        var node = &self.*.nodes[term];
        if (node.*.grad) |grad| {
            // std.debug.print("Returning cached grad {s} = {?}\n", .{ node.*.name, grad });
            return grad;
        }
        const ret = switch (node.value) {
            .value => if (term == wrt) @as(f64, 1.0) else @as(f64, 0.0),
            .add => |args| self.derive(args[0], wrt) + self.derive(args[1], wrt),
            .sub => |args| self.derive(args[0], wrt) - self.derive(args[1], wrt),
            .mul => |args| blk: {
                const lhs = self.eval(args[0]);
                const dlhs = self.derive(args[0], wrt);
                const rhs = self.eval(args[1]);
                const drhs = self.derive(args[1], wrt);
                break :blk lhs * drhs + dlhs * rhs;
            },
            .div => |args| blk: {
                const lhs = self.eval(args[0]);
                const dlhs = self.derive(args[0], wrt);
                const rhs = self.eval(args[1]);
                const drhs = self.derive(args[1], wrt);
                break :blk -lhs * drhs / rhs / rhs + dlhs / rhs;
            },
            .neg => |arg| -self.derive(arg, wrt),
            .unary_fn => |args| self.derive(args.idx, wrt) * args.grad(self.eval(args.idx)),
        };
        node.*.grad = ret;
        return ret;
    }
    fn backward(self: *Tape, term: TapeIndex) void {
        var nodes = &self.*.nodes;
        nodes.*[term].grad = 1.0;
        for (0..term + 1) |i| {
            const ii = term - i;
            var node = &nodes.*[ii];
            const grad = node.*.grad orelse 0;
            switch (node.value) {
                .value => {},
                .add => |args| {
                    nodes.*[args[0]].grad = if (nodes.*[args[0]].grad) |v| v + grad else grad;
                    nodes.*[args[1]].grad = if (nodes.*[args[1]].grad) |v| v + grad else grad;
                },
                .sub => |args| {
                    nodes.*[args[0]].grad = if (nodes.*[args[0]].grad) |v| v + grad else grad;
                    nodes.*[args[1]].grad = if (nodes.*[args[1]].grad) |v| v - grad else -grad;
                },
                .mul => |args| {
                    const lhs_grad = self.eval(args[1]) * grad;
                    nodes.*[args[0]].grad = if (nodes.*[args[0]].grad) |v| v + lhs_grad else lhs_grad;
                    const rhs_grad = self.eval(args[0]) * grad;
                    nodes.*[args[1]].grad = if (nodes.*[args[1]].grad) |v| v + rhs_grad else rhs_grad;
                },
                .div => |args| {
                    const lhs = self.eval(args[0]);
                    const rhs = self.eval(args[1]);
                    const lhs_grad = -lhs * grad / rhs / rhs;
                    nodes.*[args[0]].grad = if (nodes.*[args[0]].grad) |v| v + lhs_grad else lhs_grad;
                    const rhs_grad = grad / rhs;
                    nodes.*[args[1]].grad = if (nodes.*[args[1]].grad) |v| v + rhs_grad else rhs_grad;
                },
                .neg => |arg| {
                    nodes.*[arg].grad = if (nodes.*[arg].grad) |v| v - grad else -grad;
                },
                .unary_fn => |args| {
                    const term_grad = grad * args.grad(self.eval(args.idx));
                    nodes.*[args.idx].grad = if (nodes.*[args.idx].grad) |v| v + term_grad else term_grad;
                },
            }
        }
    }
    pub fn gen_graph(self: *Tape, term: TapeIndex, wrt: TapeIndex, allocator: *std.mem.Allocator) !?TapeTerm {
        var nodes = &self.*.nodes;
        var node = &nodes.*[term];
        switch (node.value) {
            .value => |_| {
                debug_print("gen_graph[{d}] .value {d} {d}\n", .{ self.*.count, term, wrt });
                if (term == wrt) {
                    return TapeTerm.from(self, 1);
                }
            },
            .add => |args| {
                debug_print("gen_graph .add ({d} {d}) {d}\n", .{ args[0], args[1], wrt });
                const lhs = try self.gen_graph(args[0], wrt, allocator);
                const rhs = try self.gen_graph(args[1], wrt, allocator);
                if (lhs) |lhs2| {
                    if (rhs) |rhs2| {
                        debug_print("  .add both derived = ({?} {?})\n", .{ lhs2.idx, rhs2.idx });
                        return try lhs2.add(rhs2, allocator);
                    } else {
                        return lhs2;
                    }
                } else {
                    return null;
                }
            },
            .sub => |args| {
                debug_print("gen_graph .sub ({d} {d}) {d}\n", .{ args[0], args[1], wrt });
                const lhs = try self.gen_graph(args[0], wrt, allocator);
                const rhs = try self.gen_graph(args[1], wrt, allocator);
                if (lhs) |lhs2| {
                    if (rhs) |rhs2| {
                        return try lhs2.sub(rhs2, allocator);
                    } else {
                        return lhs2;
                    }
                } else {
                    return null;
                }
            },
            .mul => |args| {
                debug_print("gen_graph[{d}] .mul ({d} {d}) {d}\n", .{ self.*.count, args[0], args[1], wrt });
                const lhs = TapeTerm.from(self, args[0]);
                const dlhs_opt = try self.gen_graph(args[0], wrt, allocator);
                const rhs = TapeTerm.from(self, args[1]);
                const drhs_opt = try self.gen_graph(args[1], wrt, allocator);
                if (dlhs_opt) |dlhs| {
                    if (drhs_opt) |drhs| {
                        debug_print("  .mul both derived = ({?} {?})\n", .{ dlhs.idx, drhs.idx });
                        return try (try lhs.mul(drhs, allocator)).add(try dlhs.mul(rhs, allocator), allocator);
                    } else {
                        debug_print("  .mul lhs derived = {?}\n", .{dlhs.idx});
                        return try dlhs.mul(rhs, allocator);
                    }
                } else if (drhs_opt) |drhs| {
                    debug_print("  .mul rhs derived = {?}\n", .{drhs.idx});
                    return try lhs.mul(drhs, allocator);
                } else {
                    debug_print("  .mul none derived\n", .{});
                    return null;
                }
            },
            .div => |args| {
                debug_print("gen_graph .div ({d} {d}) {d}\n", .{ args[0], args[1], wrt });
                const lhs = TapeTerm.from(self, args[0]);
                const dlhs_opt = try self.gen_graph(args[0], wrt, allocator);
                const rhs = TapeTerm.from(self, args[1]);
                const drhs_opt = try self.gen_graph(args[1], wrt, allocator);
                if (dlhs_opt) |dlhs| {
                    if (drhs_opt) |drhs2| {
                        return try (try (try (try (try lhs.mul(drhs2, allocator)).div(rhs, allocator)).div(rhs, allocator)).neg(allocator)).add(try dlhs.div(rhs, allocator), allocator);
                    } else {
                        return try dlhs.div(rhs, allocator);
                    }
                } else if (drhs_opt) |drhs2| {
                    return try (try (try (try lhs.mul(drhs2, allocator)).div(rhs, allocator)).div(rhs, allocator)).neg(allocator);
                } else {
                    return null;
                }
            },
            .neg => |arg| {
                debug_print("gen_graph[{d}] .neg {d} {d}\n", .{ self.*.count, arg, wrt });
                return try (try self.gen_graph(arg, wrt, allocator) orelse return null).neg(allocator);
            },
            .unary_fn => |args| {
                debug_print("gen_graph[{d}] .unary_fn {d} {d}\n", .{ self.*.count, args.idx, wrt });
                const f = args.gen_graph orelse return null;
                const input_derive = try self.gen_graph(args.idx, wrt, allocator) orelse return null;
                return f(self, args.idx, input_derive.idx, term, allocator);
            },
        }
        return null;
    }
    pub fn clear_data(self: *Tape) void {
        for (self.*.nodes[0..self.*.count]) |*node| {
            node.*.data = null;
        }
    }
    pub fn clear_grad(self: *Tape) void {
        for (self.*.nodes[0..self.*.count]) |*node| {
            node.*.grad = null;
        }
    }
    pub fn dump(self: *const Tape) void {
        for (self.*.nodes[0..self.*.count], 0..) |*node, i| {
            std.debug.print("node[{d}]: {?}\n", .{ i, node.value });
        }
    }
    pub fn dot(self: *const Tape, writer: anytype) !void {
        try writer.print("digraph D {{\n", .{});
        for (self.*.nodes[0..self.*.count], 0..) |node, i| {
            try writer.print("a{d} [label=\"{s}\", shape=rect];\n", .{ i, node.name });
        }
        for (self.*.nodes[0..self.*.count], 0..) |node, i| {
            switch (node.value) {
                .value => |_| {},
                .add => |args| try writer.print("a{d} -> a{d};\na{d} -> a{d};\n", .{ args[0], i, args[1], i }),
                .sub => |args| try writer.print("a{d} -> a{d};\na{d} -> a{d};\n", .{ args[0], i, args[1], i }),
                .mul => |args| try writer.print("a{d} -> a{d};\na{d} -> a{d};\n", .{ args[0], i, args[1], i }),
                .div => |args| try writer.print("a{d} -> a{d};\na{d} -> a{d};\n", .{ args[0], i, args[1], i }),
                .neg => |arg| try writer.print("a{d} -> a{d};\n", .{ arg, i }),
                .unary_fn => |args| try writer.print("a{d} -> a{d};\n", .{ args.idx, i }),
            }
        }
        try writer.print("}}\n", .{});
    }
};

pub const TapeIndex = u32;
pub const GenGraph = *const fn (*Tape, TapeIndex, TapeIndex, TapeIndex, *Allocator) ?TapeTerm;

pub const TapeValue = union(enum) {
    value: f64,
    add: [2]TapeIndex,
    sub: [2]TapeIndex,
    mul: [2]TapeIndex,
    div: [2]TapeIndex,
    neg: TapeIndex,
    unary_fn: struct { idx: TapeIndex, f: *const fn (f64) f64, grad: *const fn (f64) f64, gen_graph: ?GenGraph },
};

test "tape_value" {
    const t = TapeValue{ .value = 123 };
    switch (t) {
        .value => |*v| try expect(v.* == 123),
        else => expect(false),
    }

    const t2 = TapeValue{ .add = .{ 0, 1 } };
    switch (t2) {
        .add => |*v| try expect(v.*[0] == 0 and v.*[1] == 1),
        else => expect(false),
    }
}

fn concatWithOp(allocator: *std.mem.Allocator, lhs: [:0]const u8, op: [:0]const u8, rhs: [:0]const u8) ![:0]const u8 {
    var bufs: [3][]const u8 = undefined;
    bufs[0] = lhs;
    bufs[1] = op;
    bufs[2] = rhs;
    return std.mem.concatWithSentinel(allocator.*, u8, &bufs, 0);
}

fn concatWithOpParen(allocator: *std.mem.Allocator, lhs: [:0]const u8, op: [:0]const u8, rhs: [:0]const u8) ![:0]const u8 {
    var bufs: [5][]const u8 = undefined;
    bufs[0] = "(";
    bufs[1] = lhs;
    bufs[2] = op;
    bufs[3] = rhs;
    bufs[4] = ")";
    return std.mem.concatWithSentinel(allocator.*, u8, &bufs, 0);
}
