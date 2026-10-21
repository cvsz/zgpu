// ============================================================================
// zGPU: Hardware Divergence Stack (IP / Active Mask Stack)
// Synthesizable SystemVerilog Module
// ============================================================================

`timescale 1ns / 1ps

module zgpu_divergence_stack #(
    parameter DEPTH = 16
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        push,
    input  wire        pop,
    input  wire [31:0] push_pc,
    input  wire [31:0] push_mask,
    output wire [31:0] top_pc,
    output wire [31:0] top_mask,
    output wire        empty,
    output wire        full
);

    reg [31:0] stack_pc   [0:DEPTH-1];
    reg [31:0] stack_mask [0:DEPTH-1];
    reg [4:0]  ptr;

    assign empty    = (ptr == 0);
    assign full     = (ptr == DEPTH);
    assign top_pc   = (ptr > 0) ? stack_pc[ptr - 1] : 32'h0;
    assign top_mask = (ptr > 0) ? stack_mask[ptr - 1] : 32'h0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ptr <= 5'h0;
        end else begin
            if (push && !full) begin
                stack_pc[ptr]   <= push_pc;
                stack_mask[ptr] <= push_mask;
                ptr <= ptr + 1;
            end else if (pop && !empty) begin
                ptr <= ptr - 1;
            end
        end
    end

endmodule
