d2(o7).
d7(o5).
d10(o8).
d6(o1).
d13(o7).
d9(o6).
d7(o6).
d8(o6).
d11(o11).
d10(o11).

d6(X) :- d0(X), not d10(X).
d3(X) :- d8(X), not d7(X).
d6(X) :- d4(X), not d9(X).
d9(X) :- d5(X), not d8(X).
d12(X) :- not d2(X), d1(X).
d13(X) :- d3(X), not d11(X).
d1(X) :- not d8(X), d13(X).
d5(X) :- not d2(X), d13(X).
d7(X) :- not d0(X), d13(X).
d7(X) :- d5(X), not d8(X).
d6(X) :- d4(X), not d8(X).
d4(X) :- d2(X), not d9(X).
d3(X) :- not d10(X), d4(X).
d0(X) :- not d1(X), d4(X).
d0(X) :- not d8(X), d11(X).