d3(o9).
d5(o6).
d7(o10).
d8(o10).
d0(o7).
d1(o7).
d13(o7).
d13(o13).
d7(o8).
d2(o12).

d12(X) :- d5(X), not d10(X), not d0(X).
d12(X) :- d3(X), not d11(X), not d8(X).
d9(X) :- d3(X), not d12(X), not d7(X).
d6(X) :- not d2(X), d4(X), not d5(X).
d11(X) :- d9(X), not d3(X), not d8(X).
d10(X) :- d0(X), not d2(X), not d5(X).
d11(X) :- not d0(X), d4(X), not d9(X).
d4(X) :- d0(X), not d2(X), not d8(X).
d10(X) :- not d0(X), d6(X), not d12(X).
d7(X) :- not d0(X), d1(X), not d4(X).
d4(X) :- d5(X), not d6(X), not d2(X).
d12(X) :- d8(X), not d9(X), not d0(X).
d9(X) :- not d1(X), d10(X), not d8(X).
d6(X) :- d3(X), not d4(X), not d13(X).
d10(X) :- d9(X), not d6(X), not d11(X).