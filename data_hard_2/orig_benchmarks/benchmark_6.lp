d10(o11).
d8(o1).
d13(o13).
d8(o16).
d0(o10).
d3(o11).
d1(o9).
d2(o13).
d1(o11).
d9(o8).

d8(X) :- d7(X), not d5(X).
d10(X) :- not d8(X), d0(X).
d9(X) :- not d4(X), d8(X).
d3(X) :- not d13(X), d4(X).
d7(X) :- d8(X), not d5(X).
d5(X) :- not d12(X), d6(X).
d8(X) :- d9(X), not d5(X).
d1(X) :- not d0(X), d8(X).
d6(X) :- not d10(X), d9(X).
d12(X) :- not d13(X), d9(X).
d11(X) :- not d8(X), d6(X).
d10(X) :- not d0(X), d11(X).
d1(X) :- not d10(X), d5(X).
d9(X) :- not d1(X), d7(X).
d13(X) :- not d4(X), d1(X).
d4(X) :- not d7(X), d3(X).
d12(X) :- not d0(X), d7(X).
d8(X) :- not d12(X), d5(X).
d13(X) :- d1(X), not d2(X).
d7(X) :- not d2(X), d5(X).