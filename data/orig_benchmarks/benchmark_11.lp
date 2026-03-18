d6(o6).
d2(o5).
d1(o8).
d2(o9).
d0(o1).
d6(o2).

d9(X) :- not d8(X), d7(X), not d0(X).
d4(X) :- not d8(X), d3(X), not d1(X).
d5(X) :- not d3(X), d7(X), not d0(X).
d8(X) :- not d6(X), d7(X), not d4(X).
d2(X) :- not d3(X), d6(X), not d1(X).
d7(X) :- not d2(X), not d6(X), d5(X).
d9(X) :- not d6(X), d7(X), not d1(X).
d0(X) :- not d7(X), not d1(X), d4(X).
d3(X) :- not d6(X), not d5(X), d4(X).
d0(X) :- not d8(X), d5(X), not d7(X).
d0(X) :- d9(X), not d5(X), not d4(X).