const std = @import("std");
const ad = @import("./ad.zig");
const Tape = ad.Tape;
const TapeTerm = ad.TapeTerm;
const TapeIndex = ad.TapeIndex;
const expect = std.testing.expect;

pub fn main() !void {
    std.debug.print("size: {?}\n", .{@sizeOf(ad.TapeNode)});
    // try gen_graph_test();
    try gaussian_higher_order_demo();
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

fn gen_graph_test() !void {
    const file = try std.fs.cwd().createFile(
        "zigdata.csv",
        .{ .read = true },
    );
    defer file.close();

    const writer = file.writer();
    try writer.print("x, f, df/dx, d2f/dx2,\n", .{});

    var aa = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer aa.deinit();
    var allocator = aa.allocator();

    var tape = Tape.new();
    const x = tape.variable("x", 0.0);
    const x2 = try x.mul(x, &allocator);
    const nx2 = try x2.neg(&allocator);
    const f_derived = try nx2.gen_graph(x, &allocator) orelse @panic("No dependence");
    const f_derived2 = try f_derived.gen_graph(x, &allocator) orelse @panic("No dependence");

    std.debug.print("x2: {?}\n", .{x2.idx});
    std.debug.print("f_derived: {?}\n", .{f_derived.idx});
    tape.dump();

    for (0..100) |i| {
        const xval = (@as(f64, @floatFromInt(i)) - 50.0) / 10.0;
        tape.clear_data();
        x.set(xval);
        const f_val = nx2.eval();
        const f_derive_val = f_derived.eval();
        const f_derive2_val = f_derived2.eval();
        try writer.print("{?}, {?}, {?}, {?}\n", .{ xval, f_val, f_derive_val, f_derive2_val });
    }
}

fn gen_graph_exp(tape: *Tape, _input: TapeIndex, output: TapeIndex, input_derive: TapeIndex, allocator: *std.mem.Allocator) ?TapeTerm {
    _ = _input;
    const output_term = TapeTerm.from(tape, output);
    const input_derive_term = TapeTerm.from(tape, input_derive);
    const result = input_derive_term.mul(output_term, allocator) catch @panic("");
    return result;
}

fn gaussian_higher_order_demo() !void {
    const file = try std.fs.cwd().createFile(
        "zigdata.csv",
        .{ .read = true },
    );
    defer file.close();

    const writer = file.writer();
    try writer.print("x, f, df/dx, d2f/dx2,\n", .{});

    var aa = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer aa.deinit();
    var allocator = aa.allocator();

    var tape = Tape.new();
    const x = tape.variable("x", 0.0);
    const x2 = try x.mul(x, &allocator);
    const nx2 = try x2.neg(&allocator);
    const gaussian = try nx2.apply(&allocator, "exp", &exp, &exp, &gen_graph_exp);
    const derived = try gaussian.gen_graph(x, &allocator) orelse @panic("derived var doesn't exist");
    const derived2 = try derived.gen_graph(x, &allocator) orelse @panic("derived var doesn't exist");
    tape.dump();
    for (0..100) |i| {
        const xval = (@as(f64, @floatFromInt(i)) - 50.0) / 10.0;
        tape.clear_data();
        x.set(xval);
        const gaussianval = gaussian.eval();
        const gaus_derive = derived.eval();
        const gaus_derive2 = derived2.eval();
        try writer.print("{?}, {?}, {?}, {?}\n", .{ xval, gaussianval, gaus_derive, gaus_derive2 });
    }
}
