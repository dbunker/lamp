d0(o3).
d2(o3).
d1(o2).
d3(o2).

d1(X) :- not d0(X), d4(X).
d4(X) :- not d2(X), d3(X).