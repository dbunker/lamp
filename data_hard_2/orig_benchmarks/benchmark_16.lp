d0(o13).
d7(o6).
d1(o13).
d16(o11).
d7(o2).
d11(o1).
d3(o13).
d9(o6).
d1(o6).
d14(o10).
d13(o9).
d12(o4).
d17(o5).
d0(o11).
d14(o12).

d7(X) :- not d17(X), d4(X), d9(X).
d14(X) :- not d7(X), d17(X), d12(X).
d0(X) :- d1(X), d3(X), not d15(X).
d9(X) :- not d13(X), d0(X), d1(X).
d12(X) :- d2(X), d16(X), not d6(X).
d8(X) :- d17(X), d7(X), not d12(X).
d10(X) :- d8(X), d16(X), not d9(X).
d0(X) :- not d8(X), d4(X), d11(X).
d8(X) :- not d7(X), d0(X), d15(X).
d12(X) :- d2(X), d0(X), not d6(X).
d15(X) :- d17(X), d10(X), not d3(X).
d17(X) :- not d13(X), d0(X), d3(X).
d12(X) :- d8(X), not d17(X), d0(X).
d2(X) :- not d3(X), d8(X), d9(X).
d5(X) :- not d4(X), d0(X), d9(X).
d7(X) :- not d8(X), d4(X), d12(X).
d3(X) :- d1(X), d7(X), not d2(X).
d12(X) :- not d16(X), d13(X), d14(X).
d4(X) :- d2(X), d16(X), not d15(X).
d6(X) :- d2(X), d17(X), not d11(X).