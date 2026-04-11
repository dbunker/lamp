d2(o3).
d7(o0).
d6(o1).
d4(o5).
d5(o0).
d7(o2).
d9(o2).
d6(o4).
d3(o2).

d0(X) :- not d9(X), d7(X).
d0(X) :- d6(X), not d9(X).
d0(X) :- d5(X), not d9(X).
d0(X) :- not d9(X), d2(X).
d6(X) :- d5(X), not d9(X).
d5(X) :- d4(X).
d5(X) :- d7(X).
d8(X) :- d9(X).
d1(X) :- d9(X), d3(X), d7(X).