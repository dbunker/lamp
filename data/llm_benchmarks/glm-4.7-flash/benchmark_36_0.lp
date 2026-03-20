d4(o4).
d6(o3).
d4(o1).
d2(o1).
d1(o3).
d3(o2).

d2(X) :- d1(X), not d4(X).
d6(X) :- d4(X).
d8(X) :- d4(X).
d5(X) :- d4(X).