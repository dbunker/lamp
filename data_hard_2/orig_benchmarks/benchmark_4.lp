d0(o7).
d8(o7).
d6(o7).
d11(o1).
d0(o2).
d2(o0).
d7(o0).
d10(o6).
d1(o9).
d5(o10).
d10(o7).
d2(o13).
d8(o12).
d11(o6).
d4(o6).

d13(X) :- d4(X), d10(X), not d3(X).
d3(X) :- not d12(X), d7(X), d5(X).
d9(X) :- d7(X), d6(X), not d2(X).
d12(X) :- d2(X), not d7(X), d3(X).
d4(X) :- not d13(X), d1(X), d7(X).
d2(X) :- not d12(X), d1(X), d5(X).
d9(X) :- d2(X), not d1(X), d0(X).
d9(X) :- not d7(X), d10(X), d6(X).
d1(X) :- d2(X), not d6(X), d13(X).
d9(X) :- not d8(X), d12(X), d5(X).
d3(X) :- d2(X), not d13(X), d9(X).
d1(X) :- d0(X), d6(X), not d10(X).
d1(X) :- d4(X), d10(X), not d3(X).
d4(X) :- d2(X), not d8(X), d11(X).
d10(X) :- d8(X), d12(X), not d9(X).
d6(X) :- d4(X), not d9(X), d0(X).
d11(X) :- d0(X), d7(X), not d12(X).
d8(X) :- d1(X), d13(X), not d12(X).
d0(X) :- d4(X), d9(X), not d2(X).
d2(X) :- d8(X), not d11(X), d9(X).