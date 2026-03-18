d11(o13).
d1(o13).
d8(o13).
d3(o6).
d5(o17).
d7(o8).
d9(o9).
d2(o8).
d13(o2).
d1(o8).

d7(X) :- not d0(X), d1(X), d9(X).
d2(X) :- not d13(X), d1(X), d3(X).
d3(X) :- not d4(X), d7(X), d13(X).
d2(X) :- not d6(X), d10(X), d7(X).
d8(X) :- not d13(X), d1(X), d10(X).
d10(X) :- not d13(X), d1(X), d7(X).
d2(X) :- d1(X), not d5(X), d0(X).
d0(X) :- not d12(X), d7(X), d11(X).
d4(X) :- d8(X), not d7(X), d1(X).
d5(X) :- d8(X), d1(X), not d6(X).
d11(X) :- d2(X), not d10(X), d6(X).
d8(X) :- d1(X), d6(X), not d12(X).
d3(X) :- not d8(X), d12(X), d6(X).
d8(X) :- d4(X), d11(X), not d12(X).
d4(X) :- not d5(X), d13(X), d11(X).
d12(X) :- not d7(X), d4(X), d1(X).
d13(X) :- d0(X), d12(X), not d10(X).
d6(X) :- d1(X), not d3(X), d11(X).
d13(X) :- not d7(X), d1(X), d10(X).
d12(X) :- not d13(X), d8(X), d10(X).