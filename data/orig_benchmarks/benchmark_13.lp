d7(o9).
d8(o4).
d5(o2).
d4(o2).
d3(o9).
d7(o6).

d2(X) :- d1(X), d8(X), not d9(X).
d3(X) :- d6(X), d7(X), not d1(X).
d2(X) :- d7(X), not d0(X), d4(X).
d5(X) :- d9(X), d0(X), not d4(X).
d0(X) :- not d6(X), d7(X), d4(X).
d9(X) :- not d8(X), d6(X), d5(X).
d9(X) :- d3(X), not d0(X), d4(X).
d6(X) :- d9(X), not d5(X), d4(X).
d4(X) :- d9(X), not d3(X), d2(X).
d9(X) :- not d8(X), d6(X), d5(X).
d1(X) :- not d8(X), d5(X), d4(X).