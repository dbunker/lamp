d11(o7).
d6(o4).
d1(o1).
d4(o7).
d15(o2).
d8(o10).
d9(o12).
d6(o8).
d3(o15).
d10(o15).

d11(X) :- d17(X), d9(X), not d15(X).
d17(X) :- d1(X), not d9(X), d0(X).
d16(X) :- not d6(X), d10(X), d13(X).
d1(X) :- not d10(X), d9(X), d5(X).
d6(X) :- not d0(X), d3(X), d11(X).
d7(X) :- not d5(X), d12(X), d3(X).
d7(X) :- not d1(X), d16(X), d15(X).
d5(X) :- not d7(X), d0(X), d14(X).
d3(X) :- d4(X), d13(X), not d16(X).
d1(X) :- not d12(X), d13(X), d6(X).
d2(X) :- not d13(X), d1(X), d10(X).
d14(X) :- d8(X), not d17(X), d4(X).
d5(X) :- d0(X), d12(X), not d14(X).
d9(X) :- not d17(X), d10(X), d15(X).
d1(X) :- not d16(X), d4(X), d15(X).
d12(X) :- d13(X), d3(X), not d15(X).
d17(X) :- d0(X), not d3(X), d11(X).
d0(X) :- d12(X), not d3(X), d5(X).
d13(X) :- not d1(X), d4(X), d11(X).
d9(X) :- d4(X), not d6(X), d17(X).