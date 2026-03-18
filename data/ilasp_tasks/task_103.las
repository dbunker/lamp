#constant(obj, o1).
#constant(obj, o4).
#constant(obj, o5).

#modeh(d0(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).

#pos(eg1, {
  d0(o5)
}, {
  d0(o1), d0(o4)
}, {
  d1(o5).
  d2(o1).
  d2(o4).
  d3(o4).
  d3(o5).
  d4(o4).
}).