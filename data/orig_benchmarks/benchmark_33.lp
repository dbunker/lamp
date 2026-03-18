d4(o1).
d5(o2).
d6(o4).
d3(o5).
d6(o5).
d1(o3).

d7(X) :- not d9(X), d5(X), d4(X).
d1(X) :- d9(X), not d6(X), d4(X).
d8(X) :- d0(X), d7(X), not d3(X).
d9(X) :- d5(X), d7(X), not d1(X).
d0(X) :- not d6(X), d3(X), d7(X).
d6(X) :- not d9(X), d7(X), d4(X).
d0(X) :- d3(X), not d4(X), d6(X).
d2(X) :- d9(X), d5(X), not d1(X).
d3(X) :- d9(X), d7(X), not d0(X).
d1(X) :- not d8(X), d5(X), d4(X).
d3(X) :- d6(X), not d5(X), d4(X).