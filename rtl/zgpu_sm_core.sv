// ============================================================================
// zGPU: 32-Lane SIMT Streaming Multiprocessor Execution Core
// Synthesizable SystemVerilog Module
// ============================================================================

`timescale 1ns / 1ps

module zgpu_sm_core #(
    parameter LANES = 32
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire [31:0] warp_active_mask,
    input  wire [6:0]  inst_opcode,
    input  wire [31:0] op_a [0:LANES-1],
    input  wire [31:0] op_b [0:LANES-1],
    input  wire [31:0] op_c [0:LANES-1],
    output wire [31:0] results [0:LANES-1],
    output wire [LANES-1:0] zero_flags
);

    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : gen_lanes
            zgpu_lane u_lane (
                .clk(clk),
                .rst_n(rst_n),
                .lane_active(warp_active_mask[i]),
                .opcode(inst_opcode),
                .op_a(op_a[i]),
                .op_b(op_b[i]),
                .op_c(op_c[i]),
                .result(results[i]),
                .zero_flag(zero_flags[i])
            );
        end
    endgenerate

endmodule
