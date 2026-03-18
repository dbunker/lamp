d0(o13).
d6(o10).
d11(o8).
d11(o17).
d13(o17).
d10(o9).
d17(o3).
d4(o5).
d2(o8).
d12(o15).
d16(o4).
d3(o4).
d9(o11).
d0(o8).
d10(o4).

d5(X) :- d2(X), not d3(X), not d8(X).
d15(X) :- not d8(X), d10(X), not d11(X).
d10(X) :- not d7(X), not d3(X), d5(X).
d16(X) :- d12(X), not d11(X), not d5(X).
d8(X) :- not d17(X), d1(X), not d12(X).
d1(X) :- not d0(X), not d3(X), d6(X).
d10(X) :- not d4(X), d17(X), not d7(X).
d9(X) :- d17(X), not d14(X), not d3(X).
d7(X) :- not d0(X), d4(X), not d3(X).
d14(X) :- d17(X), not d16(X), not d2(X).
d11(X) :- not d17(X), not d13(X), d7(X).
d6(X) :- not d13(X), d3(X), not d15(X).
d4(X) :- not d8(X), not d3(X), d6(X).
d9(X) :- not d13(X), d4(X), not d17(X).
d14(X) :- not d1(X), d0(X), not d10(X).
d5(X) :- not d1(X), d0(X), not d7(X).
d16(X) :- d2(X), not d12(X), not d9(X).
d15(X) :- d8(X), not d5(X), not d12(X).
d17(X) :- not d5(X), not d6(X), d3(X).
d17(X) :- not d1(X), not d16(X), d6(X).