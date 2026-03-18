d3(o12).
d3(o6).
d13(o8).
d1(o10).
d16(o4).
d10(o11).
d14(o3).
d15(o11).
d7(o10).
d2(o5).

d3(X) :- d10(X), not d15(X), not d9(X).
d17(X) :- d7(X), not d9(X), not d2(X).
d1(X) :- not d12(X), d9(X), not d2(X).
d5(X) :- not d10(X), d6(X), not d15(X).
d15(X) :- not d0(X), d1(X), not d11(X).
d15(X) :- not d7(X), d14(X), not d10(X).
d15(X) :- not d17(X), not d6(X), d5(X).
d8(X) :- not d13(X), d1(X), not d6(X).
d14(X) :- not d8(X), not d6(X), d11(X).
d8(X) :- not d3(X), not d9(X), d13(X).
d6(X) :- not d0(X), not d12(X), d3(X).
d6(X) :- not d4(X), d12(X), not d11(X).
d16(X) :- not d10(X), d13(X), not d2(X).
d16(X) :- not d7(X), not d3(X), d15(X).
d11(X) :- not d17(X), not d12(X), d16(X).
d11(X) :- d4(X), not d9(X), not d5(X).
d0(X) :- not d7(X), d1(X), not d12(X).
d12(X) :- not d3(X), not d1(X), d9(X).
d9(X) :- not d11(X), d7(X), not d3(X).
d4(X) :- not d13(X), d17(X), not d11(X).