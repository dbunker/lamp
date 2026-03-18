d3(o1).
d6(o3).
d2(o4).
d1(o2).
d4(o1).
d0(o2).
d8(o1).
d4(o3).
d7(o5).

d9(X) :- d2(X), d5(X), not d1(X).
d1(X) :- d0(X), d2(X), not d9(X).
d2(X) :- d6(X), d8(X), not d0(X).
d4(X) :- not d9(X), d3(X), d6(X).
d7(X) :- not d8(X), d0(X), d6(X).
d3(X) :- not d5(X), d7(X), d4(X).
d5(X) :- d3(X), not d1(X), d6(X).
d1(X) :- d0(X), not d5(X), d4(X).
d3(X) :- d9(X), d2(X), not d7(X).
d7(X) :- d3(X), d6(X), not d1(X).
d5(X) :- not d6(X), d3(X), d8(X).