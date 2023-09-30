const std = @import("std");
const expect = std.testing.expect;

pub fn main() !void {
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
    std.debug.print("{?s} = {?}\n", .{a.get_name(), a.eval()});
    std.debug.print("{?s} = {?}\n", .{b.get_name(), b.eval()});
    std.debug.print("{?s} = {?}\n", .{c.get_name(), c.eval()});
    std.debug.print("{?s} = {?}\n", .{abcd.get_name(), abcd.eval()});
    for ([_]TapeTerm{ a, b, c, d }) |x| {
        std.debug.print("d({?s})/d{?s} = {?}\n", .{abcd.get_name(), x.get_name(), abcd.derive(x)});
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
    fn eval(self: *const TapeTerm) ?f64 {
        return self.*.tape.eval(self.*.idx);
    }
    fn derive(self: *const TapeTerm, wrt: TapeTerm) ?f64 {
        self.*.tape.clear_grad();
        return self.*.tape.derive(self.*.idx, wrt.idx);
    }
    fn add(self: *const TapeTerm, rhs: TapeTerm, arena: *std.mem.Allocator) !TapeTerm {
        var tape = self.*.tape;
        const term = tape.*.count;
        tape.*.nodes[term] = TapeNode {
            .name = try concatWithOpParen(arena, tape.*.nodes[self.*.idx].name, " + ", tape.*.nodes[rhs.idx].name),
            .value = TapeValue { .add = .{self.*.idx, rhs.idx}},
            .data = null,
            .grad = null,
        };
        tape.*.count += 1;
        return TapeTerm {
            .tape = tape,
            .idx = @intCast(term),
        };
    }
    fn sub(self: *const TapeTerm, rhs: TapeTerm, arena: *std.mem.Allocator) !TapeTerm {
        var tape = self.*.tape;
        const term = tape.*.count;
        tape.*.nodes[term] = TapeNode {
            .name = try concatWithOpParen(arena, tape.*.nodes[self.*.idx].name, " - ", tape.*.nodes[rhs.idx].name),
            .value = TapeValue { .sub = .{self.*.idx, rhs.idx}},
            .data = null,
            .grad = null,
        };
        tape.*.count += 1;
        return TapeTerm {
            .tape = tape,
            .idx = @intCast(term),
        };
    }
    fn mul(self: *const TapeTerm, rhs: TapeTerm, arena: *std.mem.Allocator) !TapeTerm {
        var tape = self.*.tape;
        const term = tape.*.count;
        tape.*.nodes[term] = TapeNode {
            .name = try concatWithOp(arena, tape.*.nodes[self.*.idx].name, " * ", tape.*.nodes[rhs.idx].name),
            .value = TapeValue { .mul = .{self.*.idx, rhs.idx}},
            .data = null,
            .grad = null,
        };
        tape.*.count += 1;
        return TapeTerm {
            .tape = tape,
            .idx = @intCast(term),
        };
    }
};

pub const Tape = struct {
    nodes: [128]TapeNode,
    count: usize,
    fn new() Tape {
        return Tape {
            .nodes = undefined,
            .count = 0,
        };
    }
    fn variable(self: *Tape, name: [:0]const u8, v: f64) TapeTerm {
        const term = self.*.count;
        self.*.nodes[term] = TapeNode {
            .name = name,
            .value = TapeValue { .value = v },
            .data = null,
            .grad = null,
        };
        self.*.count += 1;
        return TapeTerm {
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
            else => @panic("Nope"),
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
            else => @panic("TODO"),
        };
        node.*.grad = ret;
        return ret;
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
};

test "tape_value" {
    const t = TapeValue { .value = 123 };
    switch (t) {
        .value => |*v| try expect(v.* == 123),
        else => expect(false),
    }

    const t2 = TapeValue { .add = .{ 0, 1 } };
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
