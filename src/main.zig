const std = @import("std");
const expect = std.testing.expect;

pub fn main() !void {
    try gaussian_demo();
}

fn expr_demo() !void {
    var da = std.heap.page_allocator;

    var aa = std.heap.ArenaAllocator.init(da);
    defer aa.deinit();
    var allocator = aa.allocator();

    var tape = Tape.new();
    const a = tape.variable("a", 42);
    const b = tape.variable("b", 24);
    const ab = try a.add(b, &allocator);
    const c = tape.variable("c", 10);
    const abc = try ab.sub(c, &allocator);
    const d = tape.variable("d", 10);
    const abcd = try abc.mul(d, &allocator);
    const e = tape.variable("e", 3);
    const abcde = try abcd.div(e, &allocator);
    std.debug.print("{?s} = {?}\n", .{ a.get_name(), a.eval() });
    std.debug.print("{?s} = {?}\n", .{ b.get_name(), b.eval() });
    std.debug.print("{?s} = {?}\n", .{ c.get_name(), c.eval() });
    std.debug.print("{?s} = {?}\n", .{ abcde.get_name(), abcde.eval() });
    for ([_]TapeTerm{ a, b, c, d, e }) |x| {
        std.debug.print("d({?s})/d{?s} = {?}\n", .{ abcde.get_name(), x.get_name(), abcde.derive(x) });
    }
}

fn sin(x: f64) f64 {
    return std.math.sin(x);
}

fn cos(x: f64) f64 {
    return std.math.cos(x);
}

fn sine_demo() !void {
    const file = try std.fs.cwd().createFile(
        "zigdata.csv",
        .{ .read = true },
    );
    defer file.close();

    const writer = file.writer();
    try writer.print("x, sin, dsin/dx\n", .{});

    var aa = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer aa.deinit();
    var allocator = aa.allocator();

    var tape = Tape.new();
    const x = tape.variable("x", 0.0);
    const x2 = try x.mul(x, &allocator);
    const sin_x = try x2.apply(&allocator, "sin", &sin, &cos);
    for (0..100) |i| {
        const xval = (@as(f64, @floatFromInt(i)) - 50.0) / 10.0;
        tape.clear_data();
        x.set(xval);
        const sin_xval = sin_x.eval();
        const dsin_xval = sin_x.derive(x);
        try writer.print("{?}, {?}, {?}\n", .{ xval, sin_xval, dsin_xval });
    }
}

fn exp(x: f64) f64 {
    return std.math.exp(x);
}

fn gaussian_demo() !void {
    const file = try std.fs.cwd().createFile(
        "zigdata.csv",
        .{ .read = true },
    );
    defer file.close();

    const writer = file.writer();
    try writer.print("x, f, df/dx (derive), df/dx (backward),\n", .{});

    var aa = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer aa.deinit();
    var allocator = aa.allocator();

    var tape = Tape.new();
    const x = tape.variable("x", 0.0);
    const x2 = try x.mul(x, &allocator);
    const nx2 = try x2.neg(&allocator);
    const gaussian = try nx2.apply(&allocator, "exp", &exp, &exp);
    for (0..100) |i| {
        const xval = (@as(f64, @floatFromInt(i)) - 50.0) / 10.0;
        tape.clear_data();
        x.set(xval);
        const gaussianval = gaussian.eval();
        const gaus_derive = gaussian.derive(x);
        gaussian.backward();
        const gaus_back = x.get_grad() orelse return;
        try writer.print("{?}, {?}, {?}, {?}\n", .{ xval, gaussianval, gaus_derive, gaus_back });
    }
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
    fn get_name(self: *const TapeTerm) ?[:0]const u8 {
        return self.*.tape.*.nodes[self.*.idx].name;
    }
    fn get_data(self: *const TapeTerm) ?f64 {
        return self.*.tape.*.nodes[self.*.idx].data;
    }
    fn get_grad(self: *const TapeTerm) ?f64 {
        return self.*.tape.*.nodes[self.*.idx].grad;
    }
    fn set(self: *const TapeTerm, val: f64) void {
        var node = &self.*.tape.*.nodes[self.*.idx];
        switch (node.*.value) {
            .value => |*data| data.* = val,
            else => return,
        }
    }
    fn eval(self: *const TapeTerm) f64 {
        return self.*.tape.eval(self.*.idx);
    }
    fn derive(self: *const TapeTerm, wrt: TapeTerm) ?f64 {
        self.*.tape.clear_grad();
        return self.*.tape.derive(self.*.idx, wrt.idx);
    }
    fn backward(self: *const TapeTerm) void {
        self.*.tape.clear_grad();
        return self.*.tape.backward(self.*.idx);
    }
    fn add(self: *const TapeTerm, rhs: TapeTerm, arena: *std.mem.Allocator) !TapeTerm {
        return self._bin_op(rhs, arena, TapeValue{ .add = .{ self.*.idx, rhs.idx } }, " - ", true);
    }
    fn sub(self: *const TapeTerm, rhs: TapeTerm, arena: *std.mem.Allocator) !TapeTerm {
        return self._bin_op(rhs, arena, TapeValue{ .sub = .{ self.*.idx, rhs.idx } }, " - ", true);
    }
    fn mul(self: *const TapeTerm, rhs: TapeTerm, arena: *std.mem.Allocator) !TapeTerm {
        return self._bin_op(rhs, arena, TapeValue{ .mul = .{ self.*.idx, rhs.idx } }, " * ", false);
    }
    fn div(self: *const TapeTerm, rhs: TapeTerm, arena: *std.mem.Allocator) !TapeTerm {
        return self._bin_op(rhs, arena, TapeValue{ .div = .{ self.*.idx, rhs.idx } }, " / ", false);
    }
    fn neg(self: *const TapeTerm, arena: *std.mem.Allocator) !TapeTerm {
        var tape = self.*.tape;
        const term = tape.*.count;
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
    fn apply(self: *const TapeTerm, arena: *std.mem.Allocator, name: [:0]const u8, f: *const fn (f64) f64, grad: *const fn (f64) f64) !TapeTerm {
        var tape = self.*.tape;
        const term = tape.*.count;
        tape.*.nodes[term] = TapeNode{
            .name = try std.mem.concatWithSentinel(arena.*, u8, &[_][:0]const u8{ name, "(", tape.*.nodes[self.*.idx].name, ")" }, 0),
            .value = TapeValue{ .unary_fn = .{ .idx = self.*.idx, .f = f, .grad = grad } },
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
};

pub const Tape = struct {
    nodes: [128]TapeNode,
    count: usize,
    fn new() Tape {
        return Tape{
            .nodes = undefined,
            .count = 0,
        };
    }
    fn variable(self: *Tape, name: [:0]const u8, v: f64) TapeTerm {
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
    fn clear_data(self: *Tape) void {
        for (self.*.nodes[0..self.*.count]) |*node| {
            node.*.data = null;
        }
    }
    fn clear_grad(self: *Tape) void {
        for (self.*.nodes[0..self.*.count]) |*node| {
            node.*.grad = null;
        }
    }
};

const TapeIndex = u32;

pub const TapeValue = union(enum) {
    value: f64,
    add: [2]TapeIndex,
    sub: [2]TapeIndex,
    mul: [2]TapeIndex,
    div: [2]TapeIndex,
    neg: TapeIndex,
    unary_fn: struct { idx: TapeIndex, f: *const fn (f64) f64, grad: *const fn (f64) f64 },
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
