d12(o9).
d9(o7).
d0(o10).
d11(o1).
d10(o1).
d1(o7).
d13(o7).
d6(o3).
d2(o1).
d1(o11).

d16(X) :- d5(X), not d10(X), not d13(X).
d3(X) :- d9(X), not d5(X), not d0(X).
d4(X) :- not d16(X), not d5(X), d13(X).
d8(X) :- not d3(X), d12(X), not d9(X).
d1(X) :- d7(X), not d10(X), not d16(X).
d7(X) :- d5(X), not d13(X), not d14(X).
d8(X) :- not d6(X), not d4(X), d16(X).
d17(X) :- not d4(X), d16(X), not d9(X).
d15(X) :- not d2(X), d17(X), not d14(X).
d13(X) :- d6(X), not d16(X), not d12(X).
d9(X) :- d5(X), not d10(X), not d1(X).
d14(X) :- d8(X), not d3(X), not d15(X).
d1(X) :- not d14(X), not d13(X), d11(X).
d5(X) :- not d15(X), d16(X), not d11(X).
d3(X) :- d0(X), not d4(X), not d2(X).