d13(o12).
d3(o13).
d12(o2).
d13(o2).
d4(o9).
d2(o13).
d2(o0).
d9(o11).
d8(o0).
d8(o8).

d13(X) :- not d1(X), d12(X), not d3(X).
d6(X) :- not d4(X), not d9(X), d13(X).
d6(X) :- d10(X), not d3(X), not d5(X).
d10(X) :- not d7(X), d9(X), not d2(X).
d7(X) :- not d12(X), not d11(X), d5(X).
d0(X) :- not d7(X), d12(X), not d5(X).
d1(X) :- not d8(X), not d12(X), d9(X).
d3(X) :- not d7(X), d9(X), not d2(X).
d6(X) :- not d0(X), d8(X), not d11(X).
d11(X) :- not d10(X), d12(X), not d3(X).
d5(X) :- not d1(X), not d9(X), d11(X).
d11(X) :- not d13(X), d4(X), not d10(X).
d5(X) :- not d13(X), d7(X), not d3(X).
d7(X) :- not d0(X), d4(X), not d3(X).
d13(X) :- d12(X), not d6(X), not d2(X).
d13(X) :- d8(X), not d3(X), not d12(X).
d13(X) :- d2(X), not d0(X), not d11(X).
d4(X) :- d2(X), not d13(X), not d8(X).
d8(X) :- not d0(X), not d7(X), d11(X).
d3(X) :- not d1(X), d4(X), not d0(X).