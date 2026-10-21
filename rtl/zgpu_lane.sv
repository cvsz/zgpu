// ============================================================================
// zGPU: 32-Lane SIMT Execution Lane (ALU / FPU / Operand Mux)
// Synthesizable SystemVerilog Module
// ============================================================================

`timescale 1ns / 1ps

module zgpu_lane (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        lane_active,
    input  wire [6:0]  opcode,
    input  wire [31:0] op_a,
    input  wire [31:0] op_b,
    input  wire [31:0] op_c,
    output reg  [31:0] result,
    output reg         zero_flag
);

    // Opcodes matching zISA
    localparam OP_NOP   = 7'h00;
    localparam OP_VADD  = 7'h01;
    localparam OP_VSUB  = 7'h02;
    localparam OP_VMUL  = 7'h03;
    localparam OP_VAND  = 7'h04;
    localparam OP_VOR   = 7'h05;
    localparam OP_VXOR  = 7'h06;
    localparam OP_VFADD = 7'h10;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result    <= 32'h0;
            zero_flag <= 1'b0;
        end else if (lane_active) begin
            case (opcode)
                OP_VADD:  result <= op_a + op_b;
                OP_VSUB:  result <= op_a - op_b;
                OP_VMUL:  result <= op_a * op_b;
                OP_VAND:  result <= op_a & op_b;
                OP_VOR:   result <= op_a | op_b;
                OP_VXOR:  result <= op_a ^ op_b;
                default:  result <= 32'h0;
            endcase
            zero_flag <= (result == 32'h0);
        end
    end

endmodule
